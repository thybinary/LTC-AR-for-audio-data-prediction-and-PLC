import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# --- 0. Seaborn Theming ---
sns.set_theme(style="whitegrid", context="talk")
sns.despine()

# --- 1. Load Model (Updated to match training script) ---
chunk_size, seq_length = 512, 16
wiring = AutoNCP(256, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure we have nn imported
import torch.nn as nn

class StatefulCfC(CfC):
    def __init__(self, input_size, wiring, proj_size=None, **kwargs):
        super().__init__(input_size, wiring, proj_size=proj_size, **kwargs)
        self.mask_embedding = nn.Parameter(torch.randn(input_size))
        self.hidden_registry = {}
        self.current_seq_id = None
        
    def forward(self, x, hidden=None, mask_indicator=None):
        # Apply mask embedding scaled by the mask indicator - critical for reconstruction
        if mask_indicator is not None:
            mask_scale = mask_indicator.mean(dim=-1, keepdim=True)
            x = x + mask_scale * self.mask_embedding
        
        # Handle hidden state registry
        if hidden is None and self.current_seq_id in self.hidden_registry:
            hidden = self.hidden_registry[self.current_seq_id].to(x.device, non_blocking=True)
            
        output, new_hidden = super().forward(x, hidden)
        
        # Store hidden state
        if self.current_seq_id is not None:
            self.hidden_registry[self.current_seq_id] = new_hidden.detach().cpu()
            
        return output, new_hidden
    
    def clear_hidden_states(self):
        self.hidden_registry.clear()

model = StatefulCfC(input_size=chunk_size, wiring=wiring, proj_size=chunk_size).to(device)
state_dict = torch.load('/Volumes/Seagate/submitting job_UBCARC/Trained with 4000 audio files/model_epoch_44.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # set to evaluation mode

# --- 2. Read & Prep Test Audio ---
input_file = '/Users/yashvardhanjoshi/Downloads/1980s-Casio-Organ-C5_original.wav'
sr, audio = read_wav(input_file)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # convert to mono
audio = audio.astype(np.float32)
audio /= (np.max(np.abs(audio)) or 1.0)  # normalize to [-1,1]

segment = audio[: chunk_size * seq_length]
seq = segment.reshape(seq_length, chunk_size)
seq_t = torch.tensor(seq, device=device).unsqueeze(0)  # shape (1, seq_length, chunk_size)

# --- 3. Packet Loss Simulation & Reconstruction (SIMPLIFIED) ---
# Let's create a very simple mask pattern for easier debugging
mask = torch.ones_like(seq_t)
# Create specific packet drops (3 distinct regions)
mask[0, 3, 100:200] = 0  # Drop packet in chunk 3
mask[0, 5, 150:250] = 0  # Drop packet in chunk 5
mask[0, 7, 200:300] = 0  # Drop packet in chunk 7

# Apply mask to create corrupted signal
corrupted = seq_t * mask

# Create mask indicator properly (1 where data is missing, 0 where data exists)
mask_indicator = 1.0 - mask

# Assign a unique sequence ID for the model to track hidden states
import hashlib
seq_id = hashlib.sha256(seq_t.cpu().numpy().tobytes()).hexdigest()
model.current_seq_id = seq_id

# --- 4. Run model and collect detailed info ---
with torch.no_grad():
    hidden = None
    chunk_data = []
    outputs = []
    
    for t in range(seq_length):
        # Process each chunk, passing the mask indicator
        x_t = corrupted[:, t]
        mask_ind_t = mask_indicator[:, t]
        
        out, hidden = model(x_t, hidden, mask_indicator=mask_ind_t)
        outputs.append(out)
        
        # Save detailed info for this chunk
        chunk_info = {
            "chunk_idx": t,
            "original": seq_t[0, t].cpu().numpy(),
            "corrupted": corrupted[0, t].cpu().numpy(),
            "mask": mask[0, t].cpu().numpy(),
            "mask_indicator": mask_ind_t[0].cpu().numpy(),
            "prediction": out[0].cpu().numpy()
        }
        chunk_data.append(chunk_info)

# --- 5. Create full reconstruction for waveform visualization ---
reconstruction = torch.cat([out.unsqueeze(1) for out in outputs], dim=1)

# --- 6. Visualize reconstruction quality ---
# A. Plot full waveform with missing segments highlighted
plt.figure(figsize=(15, 6))

# Flatten tensors for easier plotting
flat_orig = seq_t.cpu().numpy().flatten()
flat_recon = reconstruction.cpu().numpy().flatten()
flat_mask = mask.cpu().numpy().flatten()

time_axis = np.arange(len(flat_orig)) / sr
plt.plot(time_axis, flat_orig, label='Original', color='blue', linewidth=1.5)
plt.plot(time_axis, flat_recon, label='Reconstructed', color='orange', linewidth=1.5)

# Highlight missing packets
missing_regions = np.where(flat_mask == 0)[0]
if len(missing_regions) > 0:
    # Group consecutive indices into regions
    region_starts = []
    region_ends = []
    current_start = missing_regions[0]
    for i in range(1, len(missing_regions)):
        if missing_regions[i] != missing_regions[i-1] + 1:
            region_starts.append(current_start)
            region_ends.append(missing_regions[i-1])
            current_start = missing_regions[i]
    region_starts.append(current_start)
    region_ends.append(missing_regions[-1])
    
    # Plot missing regions
    for start, end in zip(region_starts, region_ends):
        plt.axvspan(time_axis[start], time_axis[end], alpha=0.3, color='green', label='_nolegend_')
        
    # Only plot these points once to avoid duplicate legend entries
    plt.scatter(time_axis[missing_regions[0]], flat_orig[missing_regions[0]], 
                color='green', marker='x', label='Missing Packets')
    plt.scatter(time_axis[missing_regions[0]], flat_recon[missing_regions[0]], 
                color='orange', marker='o', label='Filled Packets')

plt.title('Waveform with Missing & Filled Packet Markers')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('waveform_reconstruction.png', dpi=300)
plt.show()

# --- 7. Plot error in reconstruction specifically at missing packets ---
plt.figure(figsize=(15, 4))
error = np.abs(flat_orig - flat_recon)
error_masked = np.copy(error)
error_masked[flat_mask == 1] = 0  # Only show errors at missing packets

plt.bar(time_axis, error_masked, width=1/sr, color='purple', alpha=0.7)
plt.title('Reconstruction Error at Missing Packets')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('reconstruction_error.png', dpi=300)
plt.show()

# --- 8. Detailed analysis of specific chunks ---
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
chunk_indices = [3, 5, 7]  # The chunks where we introduced packet loss

for i, chunk_idx in enumerate(chunk_indices):
    ax = axes[i]
    chunk = chunk_data[chunk_idx]
    
    # Sample indices for this chunk
    samples = np.arange(chunk_size)
    
    # Plot original data
    ax.plot(samples, chunk["original"], label="Original", color="blue")
    
    # Plot corrupted data
    ax.plot(samples, chunk["corrupted"], label="Corrupted", color="gray", alpha=0.5)
    
    # Plot model prediction
    ax.plot(samples, chunk["prediction"], label="Model Prediction", color="red")
    
    # Mark missing regions
    missing_idx = np.where(chunk["mask"] == 0)[0]
    if len(missing_idx) > 0:
        ax.axvspan(missing_idx[0], missing_idx[-1], alpha=0.2, color='yellow', label="Missing Region")
    
    ax.set_title(f"Chunk {chunk_idx}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()

plt.tight_layout()
plt.savefig('chunk_analysis.png', dpi=300)
plt.show()

# --- 9. Create improved scatter plot of original vs prediction values ---
# Use a better approach to collect data points
all_orig = []
all_pred = []
all_masked = []

# First collect and analyze all the masked points
for chunk in chunk_data:
    missing_idx = np.where(chunk["mask"] == 0)[0]
    if len(missing_idx) > 0:
        all_orig.extend(chunk["original"][missing_idx])
        all_pred.extend(chunk["prediction"][missing_idx])
        all_masked.extend([True] * len(missing_idx))

# Then add a controlled sample of non-masked points
for chunk in chunk_data:
    valid_idx = np.where(chunk["mask"] == 1)[0]
    if len(valid_idx) > 0:
        # Take a smaller random sample for clarity
        sample_idx = np.random.choice(valid_idx, min(50, len(valid_idx)), replace=False)
        all_orig.extend(chunk["original"][sample_idx])
        all_pred.extend(chunk["prediction"][sample_idx])
        all_masked.extend([False] * len(sample_idx))

# Convert to numpy for plotting
all_orig = np.array(all_orig)
all_pred = np.array(all_pred)
all_masked = np.array(all_masked)

# Create scatter plot with enhanced visuals
plt.figure(figsize=(10, 10))

# Plot non-masked points first (so masked points appear on top)
plt.scatter(all_orig[~all_masked], all_pred[~all_masked], 
           alpha=0.6, color='blue', label='Non-masked samples', s=30, edgecolor='none')

# Plot masked points with higher emphasis
plt.scatter(all_orig[all_masked], all_pred[all_masked],
           alpha=0.8, color='red', label='Masked samples', s=40, edgecolor='black', linewidth=0.5)

# Add identity line (y=x)
min_val = min(all_orig.min(), all_pred.min())
max_val = max(all_orig.max(), all_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')

# Add best fit lines
if np.sum(all_masked) > 0:
    from scipy.stats import linregress
    
    # Best fit for masked points
    slope_masked, intercept_masked, r_value_masked, _, _ = linregress(all_orig[all_masked], all_pred[all_masked])
    plt.plot([min_val, max_val], 
             [min_val*slope_masked + intercept_masked, max_val*slope_masked + intercept_masked],
             'r-', label=f'Masked fit (y={slope_masked:.2f}x+{intercept_masked:.2f}), r={r_value_masked:.2f}')
    
    # Best fit for non-masked points (if enough points)
    if np.sum(~all_masked) > 5:
        slope_valid, intercept_valid, r_value_valid, _, _ = linregress(all_orig[~all_masked], all_pred[~all_masked])
        plt.plot([min_val, max_val], 
                 [min_val*slope_valid + intercept_valid, max_val*slope_valid + intercept_valid],
                 'b-', label=f'Non-masked fit (y={slope_valid:.2f}x+{intercept_valid:.2f}), r={r_value_valid:.2f}')

plt.xlabel('Original Value', fontsize=14)
plt.ylabel('Model Prediction', fontsize=14)
plt.title('Original vs. Model Prediction', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.axis('equal')
plt.tight_layout()
plt.savefig('prediction_scatter.png', dpi=300)
plt.show()

# --- 10. Compute and print detailed metrics ---
print("\n===== RECONSTRUCTION ANALYSIS =====")

# Overall correlation
overall_corr = np.corrcoef(all_orig, all_pred)[0,1]
print(f"Overall correlation: {overall_corr:.4f}")

# Analyze masked points
if np.sum(all_masked) > 0:
    masked_corr = np.corrcoef(all_orig[all_masked], all_pred[all_masked])[0,1]
    masked_mse = np.mean((all_orig[all_masked] - all_pred[all_masked])**2)
    masked_mae = np.mean(np.abs(all_orig[all_masked] - all_pred[all_masked]))
    print(f"\nMasked Samples:")
    print(f"  Correlation: {masked_corr:.4f}")
    print(f"  MSE: {masked_mse:.6f}")
    print(f"  MAE: {masked_mae:.6f}")
    
    # Check if there's a negative correlation (model might be learning inverted patterns)
    if masked_corr < -0.3:
        print("\nWARNING: Detected negative correlation in masked samples!")
        inverted_pred = -all_pred[all_masked]
        inv_corr = np.corrcoef(all_orig[all_masked], inverted_pred)[0,1]
        print(f"  Correlation after inversion: {inv_corr:.4f}")
        
        # If inverting helps, suggest a scaling factor
        if inv_corr > 0.3:
            from scipy.optimize import minimize
            
            def scale_error(factor):
                return np.mean((all_orig[all_masked] - inverted_pred * factor)**2)
            
            result = minimize(scale_error, 1.0)
            best_scale = result.x[0]
            
            print(f"  Optimal scaling factor after inversion: {best_scale:.4f}")
            print(f"  Recommended transformation: prediction = -model_output * {best_scale:.4f}")
            
            # Calculate improvement with this transformation
            transformed_mse = np.mean((all_orig[all_masked] - inverted_pred * best_scale)**2)
            print(f"  MSE after transformation: {transformed_mse:.6f} (vs. original {masked_mse:.6f})")

# Analyze non-masked points
if np.sum(~all_masked) > 0:
    non_masked_corr = np.corrcoef(all_orig[~all_masked], all_pred[~all_masked])[0,1]
    non_masked_mse = np.mean((all_orig[~all_masked] - all_pred[~all_masked])**2)
    non_masked_mae = np.mean(np.abs(all_orig[~all_masked] - all_pred[~all_masked]))
    print(f"\nNon-Masked Samples:")
    print(f"  Correlation: {non_masked_corr:.4f}")
    print(f"  MSE: {non_masked_mse:.6f}")
    print(f"  MAE: {non_masked_mae:.6f}")

# ----- NEW CODE: Save the reconstructed audio -----
# 1. Create three different output files for comparison
output_basename = input_file.rsplit('.', 1)[0]  # Remove extension

# Original audio
original_output = output_basename + "_original.wav"
write_wav(original_output, sr, (flat_orig * 32767).astype(np.int16)) #16 bit Pulse Code Modulation (PCM)
print(f"\nSaved original audio as: {original_output}")

# Corrupted audio (with missing packets)
corrupted_audio = flat_orig.copy()
corrupted_audio[flat_mask == 0] = 0  # Set missing packets to zero
corrupted_output = output_basename + "_corrupted_new.wav"
write_wav(corrupted_output, sr, (corrupted_audio * 32767).astype(np.int16))
print(f"Saved corrupted audio as: {corrupted_output}")

# Reconstructed audio
reconstructed_output = output_basename + "_reconstructed_new.wav"
write_wav(reconstructed_output, sr, (flat_recon * 32767).astype(np.int16))
print(f"Saved reconstructed audio as: {reconstructed_output}")

# 2. Create a combined comparison audio for easy A/B testing
# This will play: 5 seconds of original, 1 second silence, 5 seconds of corrupted, 
# 1 second silence, 5 seconds of reconstructed
comparison_length = min(len(flat_orig), sr * 5)  # Use up to 5 seconds
silence = np.zeros(sr)  # 1 second of silence

comparison_audio = np.concatenate([
    flat_orig[:comparison_length],             # Original
    silence,                                   # Silence
    corrupted_audio[:comparison_length],       # Corrupted
    silence,                                   # Silence
    flat_recon[:comparison_length]             # Reconstructed
])

comparison_output = output_basename + "_comparison_new.wav"
write_wav(comparison_output, sr, (comparison_audio * 32767).astype(np.int16))
print(f"Saved audio comparison as: {comparison_output} (Original → Corrupted → Reconstructed)")

print("\nAudio Output Summary:")
print("1. Original audio")
print("2. Corrupted audio (with missing packets)")
print("3. Reconstructed audio (model's prediction)")
print("4. Comparison file with all three versions in sequence")