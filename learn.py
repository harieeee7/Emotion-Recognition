import os
import librosa
import numpy as np
import torch
import pickle
from shutil import move

# --- Config (change paths accordingly) ---
MODEL_PATH = 'models/audio_model.pth'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
AUDIO_DIR = 'audio_files'  # Folder with all unlabeled wav files
OUTPUT_DIR = 'sorted_audio'  # Folder to move classified files
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LENGTH = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model and label encoder ---
class AudioClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioClassifier, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(hidden_size*2, hidden_size//2, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size//2, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, num_classes)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0) + 1e-9
    return (mfcc - mean) / std

def pad_features(mfcc, max_len):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    pad_width = max_len - mfcc.shape[0]
    return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# Initialize model
model = AudioClassifier(N_MFCC, 128, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process files
for filename in os.listdir(AUDIO_DIR):
    if not filename.endswith('.wav'):
        continue
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        if len(audio) < sr:
            print(f"Skipping {filename}: less than 1 second long")
            continue
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = normalize_mfcc(mfcc.T)
        mfcc = pad_features(mfcc, MAX_PAD_LENGTH)
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # batch dim

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            pred_label = label_encoder.inverse_transform([pred_idx.item()])[0]

        print(f"{filename} --> {pred_label} (confidence: {confidence.item():.2f})")

        # Make destination folder for label
        dest_dir = os.path.join(OUTPUT_DIR, pred_label)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, filename)
        move(filepath, dest_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Classification and moving files done!")
