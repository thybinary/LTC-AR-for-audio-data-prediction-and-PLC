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
    def __init__(self, input_size, wiring, **kwargs):
        super().__init__(input_size, wiring, **kwargs)
        self.hidden_registry = {}
        self.current_seq_id = None

    def forward(self, x, hidden=None):
        if hidden is None and self.current_seq_id in self.hidden_registry:
            hidden = self.hidden_registry[self.current_seq_id].to(x.device, non_blocking=True)
        
        output,new_hidden = super().forward(x,hidden)
        return output, new_hidden

   # def clear_hidden_states(self):
        """Clear hidden states to free memory"""
       # self.hidden_registry.clear()

def simulate_packet_loss(data, loss_rate, packet_size=100):
    """GPU-only packet loss simulation"""
    batch_size, seq_len = data.shape[0], data.shape[1]
    mask = torch.ones_like(data, device=data.device)
    
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
        
def train_sequence(model, scaler, sequence, seq_ids, criterion, optimizer, device, grad_accumulation_steps=4):
    """Memory-optimized training step with gradient accumulation"""
    model.module.current_seq_id = seq_ids[0]
    
    # Initialize hidden state
    hidden = None
    if model.module.current_seq_id in model.module.hidden_registry:
        hidden = model.module.hidden_registry[model.module.current_seq_id].to(device)
    
    # Move data to GPU - explicitly specify device (hmmm something looks phishy)
    sequence = sequence.to(device, non_blocking=True).unsqueeze(-1)
    batch_size, seq_len = sequence.size(0), sequence.size(1)
    
    # Only zero gradients at the start of accumulation steps
    if hasattr(train_sequence, 'acc_step_counter'):
        train_sequence.acc_step_counter += 1
    else:
        train_sequence.acc_step_counter = 0
        
    if train_sequence.acc_step_counter % grad_accumulation_steps == 0:
        optimizer.zero_grad(set_to_none= True)
    
    
    # Process sequence in smaller chunks to save memory
    with torch.cuda.amp.autocast():
        corrupted, mask = simulate_packet_loss(sequence, 0.1)
        outputs = []
        
        
        # Batch forward passes over time steps , a change was made here from "sequence.size(1)" to "seq_len"
        for t in range(seq_len):
            output, hidden = model(corrupted[:, t], hidden)
            outputs.append(output)
        
        # Combine outputs and calculate loss
        outputs = torch.stack(outputs, dim=1)
        loss = criterion(outputs * mask, sequence * mask) #calculates loss only on the masked regions 
    
            
         # Detach hidden state. This is what needs to be optimised cause commenting this out leads to "leaked semaphore objects"
        #hidden = hidden.detach()
    
    # Scale loss by accumulation steps
    scaled_loss = loss / grad_accumulation_steps
    
    # Backpropagate with scaler. Other alt: using Truncated Backpropagation Through Time (TBPTT): breaks down the sequence into smaller chunks and performs backpropagation only over these chunks, effectively truncating the backpropagation process
    #scaler.scale(scaled_loss).backward(retain_graph=True) #keeping the graph alive by not freeing it up 
    # Only update parameters at the end of accumulation steps or what we can do is "if this is not the final accumulation step, retain the graph".
    if (train_sequence.acc_step_counter + 1) % grad_accumulation_steps == 0:
        scaler.scale(scaled_loss).backward() #Final step in accumulation: no need to backward pass 
        
        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        scaler.step(optimizer)
        scaler.update()
        
        #conditional detachment: truncating the graph aka TBPTT
        hidden=hidden.detach()
    else:
        scaler.scale(scaled_loss).backward(retain_graph=True)
    
    # Save hidden state to CPU to free GPU memory - explicitly move to CPU. We are going make a conditional move of the hidden states from the GPU to CPU: when GPU mem exceeds 80% automatically move them to the CPU
    if torch.cuda.memory_reserved(device) / torch.cuda.get_device_properties(device).total_memory < 0.8:
         model.module.hidden_registry[model.module.current_seq_id] = hidden
    else: 
        model.module.hidden_registry[model.module.current_seq_id] = hidden.detach().cpu()
    
    # Force garbage collection
    #del hidden, output, corrupted, mask
    #torch.cuda.empty_cache()
    
    #loss_value = loss.item()
    return loss.item()

def main():
    # Initialize distributed processing
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # Set device before creating tensors
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Process rank: {dist.get_rank()}, using GPU: {local_rank}")
    
    # Configuration
    input_folder = "/arc/project/st-mthorogo-1/CfC/corpus"
    chunk_size = 512  # Reduced for memory savings
    seq_length = 3  # Reduced sequence length
    num_epochs = 50
    batch_size = 16  # Reduced batch size
    grad_accumulation_steps = 4  # Accumulate gradients over 4 steps (effective batch size: 16)
    
    # Model setup - reduced size
    wiring = AutoNCP(64, 1) 
    model = StatefulCfC(1, wiring).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Dataset and DataLoader
    dataset = AudioDataset(
        input_folder, 
        chunk_size, 
        seq_length, 
        max_files=3968  # Limit number of files for testing
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,  # Reduced worker count
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
            loss = train_sequence(model, scaler, sequences, seq_ids, criterion, optimizer, device, grad_accumulation_steps)
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
                torch.save(model.module.state_dict(), f"model_epoch_{epoch}.pth")
    
    if dist.get_rank() == 0:
        print("Training completed successfully!")
    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()