# Recurrent Neural Network
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa
import sys
from os.path import isdir, exists
from os import listdir

def generate(model, impulse, output_sequence_length, x_frames):
    predicted_magnitudes = impulse
    random_chance = 0.2
    impulse_length = impulse.shape[1]

    #Generate continuation from impulse
    for _ in range(output_sequence_length):
        prediction = model(impulse.unsqueeze(0))
        predicted_magnitudes = torch.cat((predicted_magnitudes, prediction.transpose(0,1)), dim=1)
        impulse = predicted_magnitudes[:,-output_sequence_length:]
        if (np.random.random_sample() < random_chance) :
            np.random.seed()
            random_index = np.random.randint(0, (len(x_frames) - 1))                                                                                                                    
            impulse = x_frames[random_index]

    #Trim end to provide next impulse
    next_impulse = predicted_magnitudes[:, -impulse_length:]

    #Drop the original impulse from the returned audio
    predicted_magnitudes = predicted_magnitudes.detach().numpy()[:, impulse_length:]
    
    #Convert to audio
    audio = librosa.griffinlim(predicted_magnitudes)
    
    return audio, next_impulse

def preprocess_data(path, n_fft=2048,hop_length=512, win_length=2048, sequence_length = 40, sr = 44100):
    cached_x_path = path + '_x_frames.npy'
    cached_y_path = path + '_y_frames.npy'
    if exists(cached_x_path) and exists(cached_y_path):
        x_frames = np.load(cached_x_path)
        y_frames = np.load(cached_y_path)
        print("loading cached data")
        return torch.tensor(x_frames), torch.tensor(y_frames)
    
    x = [0]
    if not isdir(path):
        x, sr = librosa.load(path, sr=sr) 
    else:
        files = listdir(path)
        x = np.array([0])
        for file in files:
            if not ".DS" in file:
                audio, sr, = librosa.load(path + file, sr = 44100)
                x = np.concatenate((x, audio))
    x = np.array(x, dtype=np.float32) 
    data_tf = torch.tensor(x)
    # Compute STFT
    n = torch.stft(data_tf, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                window=torch.hann_window(win_length), center=True, normalized=False, onesided=True, return_complex=True)

    magnitude_spectrograms = torch.abs(n)
    print(data_tf.shape, n.shape, magnitude_spectrograms.shape)

    start = 0
    end = magnitude_spectrograms.shape[1] - sequence_length - 1 
    step = 1
    x_frames = []
    y_frames = []
    
    for i in range(start, end, step):
        done = int((float(i) / float(end)) * 100.0)
        sys.stdout.write('{}% data generation complete.   \r'.format(done))
        sys.stdout.flush()
        x = magnitude_spectrograms[:, i:i + sequence_length]
        y = magnitude_spectrograms[:, i + sequence_length]
        x_frames.append(x)
        y_frames.append(y)

    x_frames = torch.stack(x_frames)
    y_frames = torch.stack(y_frames)
    print(x_frames.shape, y_frames.shape)
    np.save(cached_x_path, x_frames)
    np.save(cached_y_path, y_frames)
    return x_frames, y_frames

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
    
        self.batch_norm = nn.BatchNorm1d(input_size)
        print(input_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.batch_norm(x) # BatchNorm expects [batch, features, seq_len]
        x, _ = self.lstm(x.transpose(1, 2))  # lstm expects [batch, seq_len, features]
        x = self.fc(x[:, -1, :]) 
        return x
    
class SpectrogramDataset(Dataset):
    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames

    def __len__(self):
        return self.x_frames.shape[0]  # Number of frames

    def __getitem__(self, idx):
        return self.x_frames[idx], self.y_frames[idx]