import torch
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import write as write_wav, read as read_wav
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


"The goal of this code is to create a system that can learn to predict missing portions of audio data in real-time"
"The model learns from the available data and attempts to restore lost packets to reconstruct the original audio as accurately as possible." 

# Function to simulate packet loss
def simulate_packet_loss(data, loss_rate):
    mask = np.random.choice([0, 1], size=data.shape, p=[loss_rate, 1 - loss_rate])
    return data * mask, mask

# Define the custom dataset for loading audio data. Processes the audio data in manageable chunks and normalises it from -1 to 1.
class AudioDataset(Dataset):
    def __init__(self, file_path, chunk_size):
        self.sample_rate, self.audio_data = read_wav(file_path)
        if self.audio_data.ndim > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)  # Convert to mono if stereo
        self.audio_data = self.audio_data.astype(np.float32) / np.iinfo(self.audio_data.dtype).max
        self.chunk_size = chunk_size
        self.audio_len = len(self.audio_data)
        self.num_chunks = int(np.ceil(self.audio_len / chunk_size))
        print(f"Total audio length: {self.audio_len}, Chunk size: {self.chunk_size}, Total chunks: {self.num_chunks}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx): #idx short for 'index'. here the indexing of the audio chunk data is taking place
        start = idx * self.chunk_size
        end = min((idx + 1) * self.chunk_size, self.audio_len)
        chunk = self.audio_data[start:end]
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        return chunk

# Define the real-time learning and prediction function
def real_time_learning_prediction(model, dataloader, output_file, criterion, optimizer, loss_rate=0.1, num_epochs=100):
    model.train()
    outputs = []
    masks = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i, input_chunks in enumerate(dataloader):
            input_chunks = input_chunks.squeeze(1)  # Remove batch dimension
            batch_size, chunk_size = input_chunks.shape
            original_input_tensor = input_chunks.unsqueeze(2)  # Add channel dimension

            input_chunks = input_chunks.numpy()
            input_chunks, mask = zip(*[simulate_packet_loss(chunk, loss_rate) for chunk in input_chunks])
            input_chunks = np.stack(input_chunks)
            mask = np.stack(mask)

            input_tensor = torch.tensor(input_chunks, dtype=torch.float32).unsqueeze(2)  # Add channel dimension

            optimizer.zero_grad()

            # Forward pass through the model
            outputs_tuple = model(input_tensor)
            if isinstance(outputs_tuple, tuple):
                output, hidden = outputs_tuple
            else:
                output = outputs_tuple

            # Compute loss
            loss = criterion(output, original_input_tensor)
            loss.backward()
            optimizer.step()

            # Use the model's output to fill in the missing packets
            output = output.detach().numpy()
            filled_chunks = np.where(mask == 0, output[:, :, 0], input_chunks)

            # Collect outputs and masks for processing
            outputs.append(filled_chunks.reshape(-1))
            masks.append(mask.reshape(-1))

            # Debugging statement for chunk processing
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Processed {i + 1}/{len(dataloader)} chunks, Loss: {loss.item()}")

    if outputs:
        outputs = np.concatenate(outputs)
        masks = np.concatenate(masks)
    else:
        print("No outputs to process.")
        return

    # Ensure outputs length matches the input length
    outputs = outputs[:len(dataloader.dataset.audio_data)]
    masks = masks[:len(dataloader.dataset.audio_data)]

    # Scale outputs from [-1, 1] to [-32767, 32767] standard 16-bit PCM (Pulse Code Modulation)
    scaled_outputs = (outputs * 32767).astype(np.int16)
    write_wav(output_file, dataloader.dataset.sample_rate, scaled_outputs)

    return outputs, masks

if __name__ == '__main__':
    # Example usage
    input_file = "/Users/yashvardhanjoshi/Desktop/Original_data_hoho (1).wav"  # Replace with your actual input file path
    output_file = "output.wav"
    chunk_size = 512 #"chunk size" refers to the number of audio samples that are grouped together and processed as a single unit or "chunk." 
    loss_rate = 0.1  # 0.1= 10% 0.2 = 20% packet loss

    # Initialize model, dataset, criterion, optimizer
    input_size = 1  # Assuming single-channel audio
    wiring = AutoNCP(28, 1)  # number of neurons, 1 output
    model = CfC(input_size, wiring)
    model = nn.DataParallel(model)  # Enable DataParallel for multi-GPU support

    dataset = AudioDataset(input_file, chunk_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=7, shuffle=False)  # Increased batch size

    criterion = nn.MSELoss()  # Mean Squared Error
    learning_rate = 0.019 # Adjust the learning rate as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Perform real-time learning and prediction
    outputs, masks = real_time_learning_prediction(model, dataloader, output_file, criterion, optimizer, loss_rate=loss_rate, num_epochs=100)

    # Save the model
    # torch.save(model.state_dict(), "trained_model_Swarmatron.pth")

    # Debugging outputs
    if outputs is not None:
        print("Processing completed successfully.")
    else:
        print("No outputs were processed.")

    # Plotting the results as line graphs with legend for missing and filled packets
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(len(dataset.audio_data)), dataset.audio_data, 'b-', label='Original Data', alpha=0.6)
    plt.scatter(np.arange(len(masks)), np.where(masks == 0, dataset.audio_data[:len(masks)], np.nan), c='g', marker='x', label='Missing Packets', alpha=0.6)
    plt.scatter(np.arange(len(masks)), np.where(masks == 0, outputs[:len(masks)], np.nan), c='orange', marker='o', label='Filled Packets', alpha=0.6)
    #plt.plot(np.arange(len(outputs)), outputs, 'r-', label='Predicted Data', alpha=0.6)
    plt.legend()
    plt.title('Comparison of Original, Predicted, and Missing Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
