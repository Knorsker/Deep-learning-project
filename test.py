from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

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
        return AudioData(self.audio_data.clone(), self.sample_rate)

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
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio_data = resample_transform(audio_data)
        signal = AudioData(audio_data,sample_rate)

        audio_data_noise, sample_rate_noise = torchaudio.load(noise_path)
        if sample_rate_noise != target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate_noise, new_freq=self.target_sample_rate)
            audio_data_noise = resample_transform(audio_data_noise)
        noise = AudioData(audio_data_noise, sample_rate_noise)

        # Convert to mono if stereo
        if signal.audio_data.shape[0] > 1:
            signal.audio_data = torch.mean(signal.audio_data, dim = -2, keepdim = True)
        if noise.audio_data.shape[0] > 1:
            noise.audio_data = torch.mean(noise.audio_data, dim = -2, keepdim = True)

        # Pad or trim to the specified length of 5 seconds
        self.fixed_length = int(signal.sample_rate*1)
        current_length = signal.audio_data.shape[1]
        if current_length < self.fixed_length:
            # Pad if the signal is shorter than the fixed length
            padding = self.fixed_length - current_length
            signal.audio_data = torch.nn.functional.pad(signal.audio_data, (0, padding))
        elif current_length > self.fixed_length:
            start_position = np.random.randint(0, max(1, signal.audio_data.shape[1] - self.fixed_length))
            signal.audio_data = signal.audio_data[:, start_position:start_position + self.fixed_length]

        self.fixed_length = int(noise.sample_rate*1) 
        current_length = noise.audio_data.shape[1]
        if current_length < self.fixed_length:
            padding = self.fixed_length - current_length
            noise.audio_data = torch.nn.functional.pad(noise.audio_data, (0, padding))
        elif current_length > self.fixed_length:
            start_position = np.random.randint(0, max(1, noise.audio_data.shape[1] - self.fixed_length))
            noise.audio_data = noise.audio_data[:, start_position:start_position + self.fixed_length]

        return signal, noise

def collate_fn(batch):
    speech, noise = zip(*batch)
    speech = [signal.audio_data for signal in speech]
    noise = [signal.audio_data for signal in noise]

    return torch.stack(speech), torch.stack(noise)

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

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

print('Dataloader - done')


import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio.functional as F
import torchaudio
from help import download, DAC, add_noise
from torch.optim.lr_scheduler import ReduceLROnPlateau

num_epochs = 100 #First: 50

# Create the model, loss function, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_sample_rate = 44100  # Replace with your desired sample rate

# Download a model
model_path = download(model_type="44khz")
model = DAC.load(model_path)

model = model.to(device).train()
criterion = nn.MSELoss() # NOTE 
lr = 1e-5 #First: 1e-4
optimizer = optim.AdamW(model.parameters(), lr=lr)

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

train_loss_vec = []
test_loss_vec = []
epoch_vec = []
snr = torch.tensor([[0]]).cuda()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for signal, noise in train_dataloader:
        signal = signal.cuda()
        noise = noise.cuda()

        # Create noisy signal
        noise += 1e-5
        noisy_signal = F.add_noise(signal, noise, snr)

        """
        # Normalize Signals
        signal_audio_data = signal.audio_data/torch.max(torch.abs(signal.audio_data))
        noisy_signal.audio_data = noisy_signal.audio_data/torch.max(torch.abs(noisy_signal.audio_data))
        """
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        noisy_signal.to(model.device)

        x = model.preprocess(noisy_signal, 44100)
        print(f'done{epoch}')
        z, codes, latents, _, _ = model.encode(x)

        # Decode audio signal
        y = model.decode(z)

        if y.shape[2] > signal.shape[2]:
            y = y[:, :, :signal.shape[2]]
        elif y.shape[2] < signal.shape[2]:
            padding = signal.shape[2] - y.shape[2]
            y = torch.nn.functional.pad(y, (0, padding))

        # Calculate the loss for the current signal
        loss = criterion(y, signal)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    # Print training statistics
    train_loss_vec.append(total_loss / len(train_dataloader))
    epoch_vec.append(epoch)
    print("------------------------------------------------------------------------")
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(train_dataloader)}')
    print("------------------------------------------------------------------------")


    # Testing
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for signals_batch in test_dataloader:
            signal = signal.cuda()
            noise = noise.cuda()
            # Create noisy signal
            noise += 1e-5
            noisy_signal = F.add_noise(signal, noise, snr)

            """
            # Normalize Signals
            signal_audio_data = signal.audio_data/torch.max(torch.abs(signal.audio_data))
            noisy_signal.audio_data = noisy_signal.audio_data/torch.max(torch.abs(noisy_signal.audio_data))
            """
            # Zero the gradients
            # optimizer.zero_grad()

            # Forward pass
            noisy_signal = noisy_signal.to(model.device)
            x = model.preprocess(noisy_signal, 44100)
            z, codes, latents, _, _ = model.encode(x)

            # Decode audio signal
            y = model.decode(z)

            if y.shape[2] > signal.shape[2]:
                y = y[:, :, :signal.shape[2]]
            elif y.shape[2] < signal.shape[2]:
                padding = signal.shape[2] - y.shape[2]
                y = torch.nn.functional.pad(y, (0, padding))

            # Calculate the loss for the current signal
            loss = criterion(y, signal)

            total_test_loss += loss.item()

    # Calculate average test loss
    average_test_loss = total_test_loss / len(test_dataloader)
    test_loss_vec.append(average_test_loss)
    
    # Print testing statistics
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Test Loss: {average_test_loss}')
    print("------------------------------------------------------------------------")
    

np.savetxt('Output_train_MSE', train_loss_vec)
np.savetxt('Output_test_MSE', test_loss_vec)    

# Save the trained model
torch.save(model.state_dict(), 'trained_MSE_model.pth')

print('Done')
