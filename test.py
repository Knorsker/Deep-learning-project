# from torch.cuda.random import set_rng_state
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
    def __init__(self, file_paths_sound, file_paths_noise, target_sample_rate=44100):
        self.file_paths_sound = file_paths_sound
        self.file_paths_noise = file_paths_noise
        self.target_sample_rate = target_sample_rate
        self.fixed_length = None

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

        # Pad or trim to the specified length of 5 seconds
        self.fixed_length = int(signal.sample_rate*0.1)
        current_length = signal.audio_data.shape[1]
        if current_length < self.fixed_length:
                # Pad if the signal is shorter than the fixed length
            padding = self.fixed_length - current_length
            signal.audio_data = torch.nn.functional.pad(signal.audio_data, (0, padding))
        elif current_length > self.fixed_length:
            start_position = np.random.randint(0, max(1, signal.audio_data.shape[1] - self.fixed_length))
            signal.audio_data = signal.audio_data[:, start_position:start_position + self.fixed_length]

        self.fixed_length = int(noise.sample_rate*0.1)
        current_length = noise.audio_data.shape[1]
        if current_length < self.fixed_length:
            padding = self.fixed_length - current_length
            noise.audio_data = torch.nn.functional.pad(noise.audio_data, (0, padding))
        elif current_length > self.fixed_length:
            start_position = np.random.randint(0, max(1, noise.audio_data.shape[1] - self.fixed_length))
            noise.audio_data = noise.audio_data[:, start_position:start_position + self.fixed_length]

        return signal, noise

def collate_fn(batch):
    return batch

# Specify the file paths as before
drive_path = r'C:\Users\julie\OneDrive\Skrivebord\Deep Learning, 02456\Project\Deep-learning-project'
audio_folder = 'Data'
noise_folder = "Noise"
file_paths_sound = [os.path.join(drive_path, audio_folder, filename) for filename in os.listdir(os.path.join(drive_path, audio_folder))]
file_paths_noise = [os.path.join(drive_path, noise_folder, filename) for filename in os.listdir(os.path.join(drive_path, noise_folder))]

# Create dataset and dataloader
audio_dataset = AudioDataset(file_paths_sound, file_paths_noise)
batch_size = 8
dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

print('done')