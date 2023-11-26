# from torch.cuda.random import set_rng_state
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split


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
        self.fixed_length = int(signal.sample_rate*5)
        current_length = signal.audio_data.shape[1]
        if current_length < self.fixed_length:
                # Pad if the signal is shorter than the fixed length
            padding = self.fixed_length - current_length
            signal.audio_data = torch.nn.functional.pad(signal.audio_data, (0, padding))
        elif current_length > self.fixed_length:
            start_position = np.random.randint(0, max(1, signal.audio_data.shape[1] - self.fixed_length))
            signal.audio_data = signal.audio_data[:, start_position:start_position + self.fixed_length]

        self.fixed_length = int(noise.sample_rate*5) 
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
# drive_path = r'C:\Users\julie\OneDrive\Skrivebord\Deep Learning, 02456\Project\Deep-learning-project'
# audio_folder = 'Data'
# noise_folder = "Noise"
drive_path = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/'
audio_folder = 'clean_fullband/vctk_wav48_silence_trimmed/p225'
noise_folder = "noise_fullband"
file_paths_sound = [os.path.join(drive_path, audio_folder, filename) for filename in os.listdir(os.path.join(drive_path, audio_folder))]
file_paths_noise = [os.path.join(drive_path, noise_folder, filename) for filename in os.listdir(os.path.join(drive_path, noise_folder))]

# Split the data into training and test sets
train_paths_sound, test_paths_sound = train_test_split(
    file_paths_sound, test_size=0.2, random_state=42
)
train_paths_noise, test_paths_noise = train_test_split(
    file_paths_noise, test_size=0.2, random_state=42
)

# Create datasets and dataloaders for training and testing
train_dataset = AudioDataset(train_paths_sound, train_paths_noise)
test_dataset = AudioDataset(test_paths_sound, test_paths_noise)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

print('Dataloader - done')


import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import dac
import torchaudio.functional as F
import torchaudio
from help import download, DAC

num_epochs = 100 #First: 50

# Create the model, loss function, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_sample_rate = 44100  # Replace with your desired sample rate

# Download a model
model_path = download(model_type="44khz")
model = DAC.load(model_path)
model = model.to(device)
criterion = nn.HuberLoss() # NOTE 
lr = 1e-5 #First: 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)

loss_vec = []
epoch_vec = []
output_train = []
output_test = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for signals_batch in train_dataloader:
        # Iterate over signals within the batch
        for signal, noise in signals_batch:
            # Calculate signal and noise power to find SNR value
            signal_power = np.sum(signal.audio_data.numpy()**2) / len(signal.audio_data.numpy())
            noise_power = np.sum(noise.audio_data.numpy()**2) / len(noise.audio_data.numpy())

            # If noise_power or L2-norm of noise is 0, correct it
            if noise_power == 0 or torch.norm(noise.audio_data, p = 2) == 0:
                noise.audio_data += 1e-5
                noise_power += 1e-5
            # Calculate signal to noise ratio
            snr_db = int(10 * np.log10(signal_power / noise_power))
            # Define noisy signal
            noisy_signal = signal
            # Create noisy signal
            noise_constant = torch.sqrt((torch.norm(signal.audio_data,p=2)**2/torch.norm(noise.audio_data,p=2)**2)*10**(-snr_db/10))
            noisy_signal.audio_data = signal.audio_data + noise_constant*noise.audio_data
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            noisy_signal.audio_data.requires_grad_(True)
            noisy_signal.to(model.device)
            noisy_signal.audio_data = torch.unsqueeze(noisy_signal.audio_data,dim=1)
            x = model.preprocess(noisy_signal.audio_data, noisy_signal.sample_rate.item())
            z, codes, latents, _, _ = model.encode(x)

            # Decode audio signal
            y = model.decode(z)

            if y.shape[2] > signal.audio_data.shape[2]:
                y = y[:, :, :signal.audio_data.shape[2]]
            elif y.shape[2] < signal.audio_data.shape[2]:
                padding = signal.audio_data.shape[2] - y.shape[2]
                y = torch.nn.functional.pad(y, (0, padding))

            # Calculate the loss for the current signal
            loss = criterion(y, signal.audio_data)

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()
        loss_vec.append(total_loss / len(train_dataloader))
        epoch_vec.append(epoch)

    # Print training statistics

    print("------------------------------------------------------------------------")
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(train_dataloader)}')
    print("------------------------------------------------------------------------")

    output_train.append(total_loss / len(train_dataloader))


    # Testing
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for signals_batch in test_dataloader:
            # Iterate over signals within the batch
            for signal, noise in signals_batch:
                # Calculate signal and noise power to find SNR value
                signal_power = np.sum(signal.audio_data.numpy()**2) / len(signal.audio_data.numpy())
                noise_power = np.sum(noise.audio_data.numpy()**2) / len(noise.audio_data.numpy())
                # If noise_power or L2-norm of noise is 0, correct it
                if noise_power == 0 or torch.norm(noise.audio_data, p=2) == 0:
                    noise.audio_data += 1e-5
                    noise_power += 1e-5
                # Calculate signal to noise ratio
                snr_db = int(10 * np.log10(signal_power / noise_power))

                # Create noisy signal
                noise_constant = torch.sqrt((torch.norm(signal.audio_data, p=2)**2 / torch.norm(noise.audio_data, p=2)**2) * 10**(-snr_db / 10))
                noisy_signal = signal
                noisy_signal.audio_data = signal.audio_data + noise_constant * noise.audio_data

                # Forward pass
                noisy_signal.to(model.device)
                noisy_signal.audio_data = torch.unsqueeze(noisy_signal.audio_data, dim=1)
                x = model.preprocess(noisy_signal.audio_data, noisy_signal.sample_rate.item())
                z, codes, latents, _, _ = model.encode(x)

                # Decode audio signal
                y = model.decode(z)

                if y.shape[2] > signal.audio_data.shape[2]:
                    y = y[:, :, :signal.audio_data.shape[2]]
                elif y.shape[2] < signal.audio_data.shape[2]:
                    padding = signal.audio_data.shape[2] - y.shape[2]
                    y = torch.nn.functional.pad(y, (0, padding))

                # Calculate the loss for the current signal
                loss = criterion(y, signal.audio_data)

                total_test_loss += loss.item()

    # Calculate average test loss
    average_test_loss = total_test_loss / len(test_dataloader)

    # Print testing statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Test Loss: {average_test_loss}')
    print("------------------------------------------------------------------------")

    output_test.append(average_test_loss)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}]')


np.savetxt('Output_train_HUBER_2', output_train)
np.savetxt('Output_test_HUBER_2', output_test)    
# Save the trained model
torch.save(model.state_dict(), 'trained_HUBER_model.pth')

print('Done')
