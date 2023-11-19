from torch.cuda.random import set_rng_state
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import torch
import torchaudio.functional as F
import numpy as np

class AudioData:
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def to(self, device):
        self.audio_data = self.audio_data.to(device)
        self.sample_rate = torch.tensor(self.sample_rate, device=device)
        return self

    def device(self):
        return self.audio_data.device

    def clone(self):
        return self.audio_data.clone()

    def audio_data(self):
        return self.audio_data

    def sample_rate(self):
        return self.sample_rate

class AudioDataset(Dataset):
    def __init__(self, file_paths_sound, file_paths_noise, target_sample_rate=44100, fixed_length=None):
        self.file_paths_sound = file_paths_sound
        self.file_paths_noise = file_paths_noise
        self.target_sample_rate = target_sample_rate
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.file_paths_sound)

    def __getitem__(self, idx):
        audio_path = self.file_paths_sound[idx]
        noise_path = np.random.choice(self.file_paths_noise)

        audio_data, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            sample_rate = target_sample_rate
        signal = AudioData(audio_data,sample_rate)

        audio_data_noise, sample_rate_noise = torchaudio.load(noise_path)
        if sample_rate_noise != target_sample_rate:
            sample_rate_noise = target_sample_rate
        noise = AudioData(audio_data_noise, sample_rate_noise)

        # Convert to mono if stereo
        if signal.audio_data.shape[0] > 1:
            signal = signal.to_mono()
        if noise.audio_data.shape[0] > 1:
            noise = noise.to_mono()

        # Pad or trim to the specified maximum length
        if self.fixed_length is not None:
            current_length = signal.audio_data.shape[1]
            if current_length < self.fixed_length:
                # Pad if the signal is shorter than the fixed length
                padding = self.fixed_length - current_length
                signal.audio_data = torch.nn.functional.pad(signal.audio_data, (0, padding))
            elif current_length > self.fixed_length:
                # Truncate if the signal is longer than the fixed length
                signal.audio_data = signal.audio_data[:, :self.fixed_length]

            current_length = noise.audio_data.shape[1]
            if current_length < self.fixed_length:
                padding = self.fixed_length - current_length
                noise.audio_data = torch.nn.functional.pad(noise.audio_data, (0, padding))
            elif current_length > self.fixed_length:
                noise.audio_data = noise.audio_data[:, :self.fixed_length]

        return signal, noise

def collate_fn(batch):
    return batch

# Specify the file paths as before
drive_path = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/'
audio_folder = 'clean_fullband/vctk_wav48_silence_trimmed/p225'
noise_folder = "noise_fullband"
file_paths_sound = [os.path.join(drive_path, audio_folder, filename) for filename in os.listdir(os.path.join(drive_path, audio_folder))]
file_paths_noise = [os.path.join(drive_path, noise_folder, filename) for filename in os.listdir(os.path.join(drive_path, noise_folder))]

max_length = 4410

# Create dataset and dataloader
audio_dataset = AudioDataset(file_paths_sound, file_paths_noise, fixed_length=max_length)
batch_size = 8
dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
"""
# Example usage
for signals in dataloader:
  for signal, noise in signals:
      print(signal.audio_data.shape)  # Print the number of frames in each signal in the batch
      print(signal.sample_rate)  # Print the sample rate of each signal in the batch
      print(noise.audio_data.shape)
      print(noise.sample_rate)
      print("----------------------")
"""

print('Done')
