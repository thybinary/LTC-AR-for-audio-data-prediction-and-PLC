import os 
# setting memory fragmentation v v imp: will split individual mem block to 128MB 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import read as read_wav
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.utils.data import DataLoader, Dataset
import glob
import hashlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import gc

class StatefulCfC(CfC):
    def __init__(self, input_size, wiring, proj_size = None, **kwargs):
        
        super().__init__(input_size, wiring, proj_size = proj_size, **kwargs)
        self.hidden_registry = {}
        self.current_seq_id = None
        self.mask_embedding = nn.Parameter(torch.randn(input_size))

    def forward(self, x, hidden=None, mask_indicator = None):
        
        # proper mask indication cause earlier it was ambiguous for the model to predict which one's masked data or not, that's why i was getting prediction in the -ves 
        if mask_indicator is None: 
            
            mask_scale = mask_indicator.mean (dim=-1, keepdim=True)
            
            #mask_indicator = torch.zeros((x.shape[0], 1), device = x.device)
        #else:
            #mask_indicator = mask_indicator.mean(dim=-1, keepdim = True)
        
        #x_with_mask = torch.cat([x, mask_indicator], dim=-1)
            x = x + mask_scale * self.mask_embedding
        
        if hidden is None and self.current_seq_id in self.hidden_registry:
            hidden = self.hidden_registry[self.current_seq_id].to(x.device, non_blocking=True)
        
        output,new_hidden = super().forward(x,hidden)
        #hidden states being properly initalised
        self.hidden_registry[self.current_seq_id]=new_hidden.detach().cpu()
        return output, new_hidden

    def clear_hidden_states(self):
        """Clear hidden states to free memory"""
        self.hidden_registry.clear()

def simulate_packet_loss(data, loss_rate, packet_size=100, fixed_pattern = False): #adding fixed pattern cause it gives the model a certain baseline to do its learning 
    """GPU-only packet loss simulation"""
    batch_size, seq_len = data.shape[0], data.shape[1]
    mask = torch.ones_like(data, device=data.device)
    
    if fixed_pattern:
        num_packets = max (1, seq_len // packet_size)
        for b in range(batch_size):
            
            for i in range (0, num_packets, 3):
                if i < num_packets: 
                    start = i * packet_size
                    end = min((i+1) * packet_size, seq_len)
                    mask[b , start:end] = 0 
    else:
    
    # Simplified packet loss simulation
        if seq_len >= packet_size:
            packet_loss = torch.rand(batch_size, max(1, seq_len // packet_size), device=data.device) < loss_rate
            for i in range(min(seq_len // packet_size, packet_loss.shape[1])):
                start = i * packet_size
                end = min((i+1) * packet_size, seq_len)
                mask[:, start:end] = ~packet_loss[:, [i]].unsqueeze(-1)
    
    return data * mask, mask

class AudioDataset(Dataset):
    def __init__(self, folder_path, chunk_size, seq_length=3, max_files=None):
        self.chunk_size = chunk_size
        self.seq_length = seq_length
        self.sequences = []
        self.sequence_ids = []
        
        total_files = 0
        total_chunks = 0
        
        files = glob.glob(os.path.join(folder_path, "*.wav")) #all the .wav files in the folder 
        if max_files:
            files = files[:max_files]

        for file_idx, file_path in enumerate(files):
            try:
                sample_rate, audio = read_wav(file_path)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Add debug info for file loading
                if dist.get_rank() == 0 and file_idx % 10 == 0:
                    print(f"\nLoading file {file_idx+1}/{len(files)}: {os.path.basename(file_path)}")
                    print(f"Sample rate: {sample_rate} Hz")
                    print(f"Duration: {len(audio)/sample_rate:.2f} seconds")

                max_val = np.max(np.abs(audio)) or 1.0
                audio = (audio.astype(np.float32) / max_val).reshape(-1)

                # Subsample audio to reduce memory usage
                num_chunks = len(audio) // chunk_size
                valid_chunks = max(0, num_chunks - self.seq_length)
                
                # Limit number of sequences per file
                max_seqs_per_file = min(100, valid_chunks)
                stride = max(1, valid_chunks // max_seqs_per_file)
                
                for i in range(0, valid_chunks, stride):
                    start = i * chunk_size
                    end = start + self.seq_length * chunk_size
                    seq = audio[start:end]
                    seq = seq.reshape(self.seq_length, chunk_size)
                    self.sequences.append(seq)
                    self.sequence_ids.append(hashlib.sha256(seq.tobytes()).hexdigest())
                    
                total_files += 1
                total_chunks += valid_chunks
                
                # Force garbage collection
                #del audio
                #gc.collect()
                
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Error loading file {file_path}: {e}")
        
        # Debugging to make sure the files are loaded
        if dist.get_rank() == 0:
            print(f"\nDataset Summary:")
            print(f"Total files loaded: {total_files}")
            print(f"Total sequences: {len(self.sequences)}")
            print(f"Total chunks: {len(self.sequences) * self.seq_length}")
            print(f"Sequence length: {self.seq_length} chunks")
            print(f"Chunk size: {chunk_size} samples")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            self.sequence_ids[idx]
        )

def collate_fn(batch):
    """Optimized collate with sequence IDs"""
    seqs, ids = zip(*batch)
    return torch.stack(seqs), list(ids)
        
def train_sequence_improved(model, scaler, sequence, seq_ids, criterion, optimizer, device, grad_accumulation_steps=4, tbptt_steps=4):
    model.module.current_seq_id = seq_ids[0]
    sequence = sequence.to(device, non_blocking=True)
    batch_size, seq_len, chunk_size = sequence.shape
    
    # Zero gradients at start of accumulation step
    if not hasattr(train_sequence_improved, "step"):
        train_sequence_improved.step = 0
    train_sequence_improved.step += 1
    
    if train_sequence_improved.step % grad_accumulation_steps == 1:
        optimizer.zero_grad(set_to_none=True)
    
    # Create packet loss simulation once for the entire sequence
    corrupted, mask = simulate_packet_loss(sequence, 0.1)
    
    mask_indicator = 1.0 - mask
    
    # More controlled TBPTT
    total_loss = 0.0
    hidden = None
    
    # Process sequence in chunks of tbptt_steps
    for t_start in range(0, seq_len, tbptt_steps):
        t_end = min(t_start + tbptt_steps, seq_len)
        chunk_outputs = []
        
        # Forward pass for this chunk
        for t in range(t_start, t_end):
            out, hidden = model(corrupted[:, t], hidden, mask_indicator=mask_indicator[:,t])
            chunk_outputs.append(out)
        
        # Stack outputs for this chunk
        chunk_out = torch.stack(chunk_outputs, dim=1)
        
        # Calculate loss for this chunk
        chunk_target = sequence[:, t_start:t_end]
        chunk_mask = mask[:, t_start:t_end]
        #chunk_loss = criterion(chunk_out * chunk_mask, chunk_target * chunk_mask)
        unmasked_loss = criterion(chunk_out * chunk_mask, chunk_target * chunk_mask)
        
        if torch.any(chunk_mask < 1.0):
            inv_mask = 1.0 - chunk_mask
            masked_loss =  criterion(chunk_out * inv_mask, chunk_target * inv_mask)
        
        #giving higher weight to masked regions 
            chunk_loss = unmasked_loss + 3.0 * masked_loss
        else:
            chunk_loss = unmasked_loss
            
        # Scale loss and backward
        scaled_loss = chunk_loss / ((seq_len + tbptt_steps - 1) // tbptt_steps) / grad_accumulation_steps
        scaler.scale(scaled_loss).backward()
        
        # Detach hidden state for next chunk
        hidden = hidden.detach()
        
        total_loss += chunk_loss.item()
    
    # Step optimizer at the end of accumulation steps
    if train_sequence_improved.step % grad_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
    
    return total_loss

def main():
    # Initialize distributed processing
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # Set device before creating tensors
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Process rank: {dist.get_rank()}, using GPU: {local_rank}")
    
    # Configuration
    input_folder = "input folder path here"
    chunk_size = 512  # Reduced for memory savings
    seq_length = 8  # Reduced sequence length
    num_epochs = 50
    batch_size = 16  # Reduced batch size
    grad_accumulation_steps = 4  # Accumulate gradients over 4 steps (effective batch size: 16)
    tbptt_steps = 4
    
    # Model setup - reduced size
    wiring = AutoNCP(256, 128) 
    model = StatefulCfC(input_size=chunk_size, wiring = wiring, proj_size = chunk_size ).to(device) # proj_size outputs a 512‑dimensional vector per time step
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Dataset and DataLoader
    dataset = AudioDataset(
        input_folder, 
        chunk_size, 
        seq_length, 
        max_files=1200  # Limit number of files for testing
    ) 
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        #drop_last=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2, 
        pin_memory=True, # pin_memory argument, which defaults to False. When using a GPU it’s better to set pin_memory=True, this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    if dist.get_rank() == 0:
        print(f"\nTraining Configuration:")
        print(f"World size: {dist.get_world_size()}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Gradient accumulation steps: {grad_accumulation_steps}")
        print(f"Effective batch size: {batch_size * grad_accumulation_steps}")
        print(f"Total batches per epoch: {len(dataloader)}")
        print(f"Sequence length: {seq_length} chunks")
        print(f"Chunk size: {chunk_size} samples")
        print(f"Total epochs: {num_epochs}\n")
    
    # Training setup - ensure criterion is on the right device
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0016, weight_decay=1e-5)
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        
        # Periodic memory cleanup
        #if epoch % 3 == 0:
           # model.module.clear_hidden_states()
            #gc.collect()
            #torch.cuda.empty_cache()
        
        for batch_idx, (sequences, seq_ids) in enumerate(dataloader):
            # Pass the device explicitly to train_sequence
            loss = train_sequence_improved(model, scaler, sequences, seq_ids, criterion, optimizer, device, grad_accumulation_steps)
            total_loss += loss
            
            # More frequent memory cleanup
            #if batch_idx % 10 == 0:
                #gc.collect()
                #torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                reserved= torch.cuda.memory_reserved(device) // 1024**2
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss:.4f} | GPU Reserved: {reserved}MB")
        
        #Force garbage collection at the end of each epoch
        #gc.collect()
        #torch.cuda.empty_cache()
        
        if dist.get_rank() == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")
            # Save model less frequently
            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), f"model_epoch_{epoch}.pth") #Instead of saving a module directly, for compatibility reasons it is recommended to instead save only its state dict.
    
    if dist.get_rank() == 0:
        print("Training completed successfully!")
    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
