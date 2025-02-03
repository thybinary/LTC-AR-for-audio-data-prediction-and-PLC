import torch
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

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize distributed processing for multi-GPU
dist.init_process_group(backend="nccl")

# Function to simulate packet loss (Now using CUDA tensors)
def simulate_packet_loss(data, loss_rate):
    mask = (torch.rand_like(torch.tensor(data, device=device)) > loss_rate).float()
    return (torch.tensor(data, device=device) * mask).cpu().numpy(), mask.cpu().numpy()

class AudioDataset(Dataset):
    def __init__(self, folder_path, chunk_size):
        self.file_paths = glob.glob(os.path.join(folder_path, "*.wav"))
        self.chunk_size = chunk_size
        self.audio_data_list = []
        self.sample_rates = []
        
        for file_path in self.file_paths:
            sample_rate, audio_data = read_wav(file_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono if stereo
            
            # Normalize audio data and ensure float32
            if np.issubdtype(audio_data.dtype, np.integer):
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            elif np.issubdtype(audio_data.dtype, np.floating):
                audio_data = audio_data.astype(np.float32)
                
            self.audio_data_list.append(audio_data)
            self.sample_rates.append(sample_rate)
            print(f"Loaded {file_path} with sample rate {sample_rate} and length {len(audio_data)}")
    
    def __len__(self):
        return sum(len(data) for data in self.audio_data_list) // self.chunk_size

    def __getitem__(self, idx): #index being requested 
        file_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size #for locating the chunk's position within the file using the remainder % of idx divided by chunk_size
        audio_data = self.audio_data_list[file_idx]
        start = local_idx * self.chunk_size
        end = min(start + self.chunk_size, len(audio_data))
        chunk = audio_data[start:end]
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        return torch.tensor(chunk, dtype=torch.float32), torch.tensor(self.sample_rates[file_idx])

# Define the real-time learning and prediction function (it is not real time learning  as we're loading the audio data)
def real_time_learning_prediction(model, dataloader, criterion, optimizer, loss_rate=0.1, num_epochs=1):
    model.train()
    outputs = []
    masks = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time ()
        
        for i, (input_chunks, sample_rate) in enumerate(dataloader):
            step_time = time.time()
            # Convert input to float32
            input_chunks = input_chunks.to(device)  # Move input data to GPU
            
            batch_size, chunk_size = input_chunks.shape
            original_input_tensor = input_chunks.unsqueeze(2).to(device)
            
            # Simulate packet loss on GPU
            input_chunks_np = input_chunks.gpu().numpy()
            input_chunks_lost, mask = zip(*[simulate_packet_loss(chunk, loss_rate) for chunk in input_chunks_np])
            input_chunks_lost = np.stack(input_chunks_lost).astype(np.float32)
            mask = np.stack(mask)
            
           # Convert back to Pytorch tensor and move to GPU
            input_tensor = torch.tensor(input_chunks_lost, dtype=torch.float32).unsqueeze(2).to(device)
            valid_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(2).to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            outputs_tuple = model(input_tensor)
            if isinstance(outputs_tuple, tuple):
                output, hidden = outputs_tuple
            else:
                output = outputs_tuple

            # Compute loss
            valid_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(2)  # Add channel dimension
            loss = criterion(output * ( 1- valid_mask), original_input_tensor * (1 - valid_mask))
            print(f"Batch {i+1} loss: {loss.item()}, Time per batch: {time.time() - step_time:.2f}s")
            
            loss.backward()
            optimizer.step()

            # Use the model's output to fill in the missing packets
            output_np = output.detach().cpu().numpy() # move ouput back to CPU for numpy processing 
            filled_chunks = np.where(mask == 0, output_np[:, :, 0], input_chunks_lost)

            # Collect outputs and masks for processing
            outputs.append(filled_chunks.reshape(-1))
            masks.append(mask.reshape(-1))
            
            print(f"Batch {i+1} loss: {loss.item()}, Time per batch: {time.time() - step_time:.2f}s")

            # Debugging statement for chunk processing
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Processed {i + 1}/{len(dataloader)} chunks, Loss: {loss.item()}")

    if outputs:
        outputs = np.concatenate(outputs)
        masks = np.concatenate(masks)
    else:
        print("No outputs to process.")
        return None, None

    return outputs, masks

if __name__ == '__main__':
    # Directory containing audio files
    input_folder = "/Users/yashvardhanjoshi/Music/SuperCollider Recordings"  # Replace with the path to your folder of audio files
    chunk_size = 512
    num_epochs = 1
    loss_rate = 0.1  # 0.1= 10% 0.2 = 20% packet loss

    # Initialize model, dataset, criterion, optimizer
    input_size = 1  # Assuming single-channel audio
    wiring = AutoNCP(28, 1)  # number of neurons, 1 output
    model = CfC(input_size, wiring).to(device)
    model = model.float() # model in float32
    model = DDP(model)  # Enable DistributedDataParallel for multi-GPU support

    dataset = AudioDataset(input_folder, chunk_size)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=6, shuffle=False)

    criterion = nn.MSELoss().to(device)  # Mean Squared Error
    learning_rate = 0.0016
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Perform real-time learning and prediction (not real-time click bait tehehehe)
    outputs, masks = real_time_learning_prediction(model, dataloader, criterion, optimizer, loss_rate=loss_rate, num_epochs=num_epochs)

    # Save the model
    torch.save(model.state_dict(), "trained_model_corpus.pth")

    # Debugging outputs
    if outputs is not None:
        print("Processing completed successfully.")
    else:
        print("No outputs were processed.")
