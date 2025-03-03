import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import read as read_wav
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.utils.data import DataLoader, Dataset
import os
import glob
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Initialize distributed processing
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

print(f"Process rank: {dist.get_rank()}, using GPU: {local_rank}")

def simulate_packet_loss(data, loss_rate, packet_size=100):
    """GPU-only packet loss simulation"""
    batch_size, seq_len, _ = data.shape
    mask = torch.ones_like(data, device=data.device)
    
    # Vectorized packet loss
    packet_loss = torch.rand(batch_size, seq_len // packet_size, device=data.device) < loss_rate
    for i in range(seq_len // packet_size):
        start = max(0, i*packet_size - 50)
        end = (i+1)*packet_size
        mask[:, start:end, :] = ~packet_loss[:, [i]].unsqueeze(-1)
    
    return data * mask, mask

class AudioDataset(Dataset):
    def __init__(self, folder_path, chunk_size, seq_length=3):
        self.chunk_size = chunk_size
        self.seq_length = seq_length
        self.sequences = []
        
        total_files = 0
        total_chunks = 0
        
        for file_idx, file_path in enumerate(glob.glob(os.path.join(folder_path, "*.wav"))):
            sample_rate, audio = read_wav(file_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Add debug info for file loading
            if dist.get_rank() == 0:
                print(f"\nLoading file {file_idx+1}: {os.path.basename(file_path)}")
                print(f"Sample rate: {sample_rate} Hz")
                print(f"Total samples: {len(audio)}")
                print(f"Duration: {len(audio)/sample_rate:.2f} seconds")

            # Handle silent audio
            max_val = np.max(np.abs(audio))
            if max_val == 0:
                audio = np.zeros_like(audio, dtype=np.float32)
                if dist.get_rank() == 0:
                    print("Warning: Silent audio file detected")
            else:
                audio = audio.astype(np.float32) / (max_val + 1e-12)

            num_chunks = len(audio) // chunk_size
            valid_chunks = num_chunks - self.seq_length
            
            if dist.get_rank() == 0:
                print(f"Chunks per file: {num_chunks}")
                print(f"Valid sequences: {valid_chunks}")

            for i in range(valid_chunks):
                start = i * chunk_size
                end = start + self.seq_length * chunk_size
                seq = audio[start:end]
                seq = seq.reshape(self.seq_length, chunk_size)
                self.sequences.append(seq)
            
            total_files += 1
            total_chunks += valid_chunks
        
        if dist.get_rank() == 0:
            print(f"\nDataset Summary:")
            print(f"Total files loaded: {total_files}")
            print(f"Total sequences: {len(self.sequences)}")
            print(f"Total chunks: {len(self.sequences) * self.seq_length}")
            print(f"Sequence length: {self.seq_length} chunks")
            print(f"Chunk size: {chunk_size} samples")
            print(f"Total training samples: {len(self.sequences) * self.seq_length * chunk_size}\n")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32) # removed device = device as Pytorch doesn't support GPU data processing. So we're going to use pin_memory = True which asynchronously sends data to the GPU and the dataloader will then expect the data to come from the CPU to GPU. 

def train_sequence(model, sequence, criterion, optimizer):
    """Process the sequence with state retention"""
    hidden = None
    #total_loss = 0
    optimizer.zero_grad()
    
    # Move entire sequence to GPU once
    sequence = sequence.to(device).unsqueeze(-1)  # [batch_size, seq_length, chunk_size, 1]
    
    losses = []
    for i, chunk in enumerate(sequence.permute(1, 0, 2, 3)):  # [seq_length, batch_size, chunk_size, 1]
        corrupted, mask = simulate_packet_loss(chunk, 0.1)
        output, hidden = model(corrupted, hidden)
        hidden = hidden.detach()
        
        loss = criterion(output * (1 - mask), chunk * (1 - mask))
        #total_loss += loss.item()
        #loss.backward(retain_graph=True)
        losses.append(loss)
        
     
    # Sum losses     
    total_loss = torch.sum(torch.stack(losses))
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping to avoid exponential loss 
    optimizer.step()
    
    return total_loss.item() / len(sequence)

if __name__ == '__main__':
    # Configuration
    input_folder = "/arc/project/st-mthorogo-1/CfC/corpus"
    chunk_size = 1024
    seq_length = 5
    num_epochs = 50
    
    # Model setup
    wiring = AutoNCP(28, 1)
    model = CfC(1, wiring).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank) #model initialised on the GPU 
    
    # Dataset and DataLoader
    dataset = AudioDataset(input_folder, chunk_size, seq_length)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    
    if dist.get_rank() == 0:
        print(f"\nTraining Configuration:")
        print(f"World size: {dist.get_world_size()}")
        print(f"Batch size per GPU: {16}")
        print(f"Total batches per epoch: {len(dataloader)}")
        print(f"Sequence length: {seq_length} chunks")
        print(f"Chunk size: {chunk_size} samples")
        print(f"Total epochs: {num_epochs}\n")
    
    # Training setup
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            #print(f"Total sequences to process: {len(dataloader)}")
            #print(f"Approx total chunks: {len(dataloader) * 8 * seq_length}")
        
        for batch_idx, sequence in enumerate(dataloader):
            loss = train_sequence(model, sequence, criterion, optimizer)
            total_loss += loss
            
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Avg Loss: {loss:.4f} | "
                      f"GPU Mem: {torch.cuda.memory_allocated(device)//1024**2}MB")
        
        if dist.get_rank() == 0:
            epoch_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f}")
            torch.save(model.module.state_dict(), f"model_epoch_{epoch}.pth")
    
    if dist.get_rank() == 0:
        print("\nTraining completed successfully!")
        print(f"Final model saved as model_epoch_{num_epochs-1}.pth")
