import os
import librosa
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LENGTH = 150
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 8

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_model.pth')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
AUDIO_DIR = 'sorted_audio/audio_files'  # Update to your dataset root folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class AudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0) + 1e-9
    return (mfcc - mean) / std

def extract_emotion_label(filename):
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }
    parts = filename.split('-')
    try:
        emotion_code = int(parts[2])
        return emotion_map.get(emotion_code, 'unknown')
    except Exception as e:
        logger.warning(f"Could not extract emotion from {filename}: {e}")
        return 'unknown'

def load_audio_files(folder_path):
    features = []
    labels = []
    logger.info(f"Loading audio files from: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                try:
                    path = os.path.join(root, file)
                    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
                    if len(audio) < sr:
                        logger.warning(f"Skipping {file} - shorter than 1 sec")
                        continue

                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                    mfcc = normalize_mfcc(mfcc.T)
                    
                    label = extract_emotion_label(file)
                    if label == 'unknown':
                        continue  # skip unknown labels

                    features.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

    logger.info(f"Total loaded features: {len(features)}")
    return features, labels

def pad_features(features, max_len=MAX_PAD_LENGTH):
    padded = []
    for mfcc in features:
        if mfcc.shape[0] > max_len:
            padded.append(mfcc[:max_len, :])
        else:
            pad_width = max_len - mfcc.shape[0]
            padded_mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            padded.append(padded_mfcc)
    return np.array(padded)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    
    features, labels = load_audio_files(AUDIO_DIR)
    if not features:
        raise ValueError("No valid audio files found! Check your dataset path and contents.")

    logger.info(f"Loaded {len(features)} samples.")
    label_counts = {label: labels.count(label) for label in set(labels)}
    logger.info(f"Label distribution: {label_counts}")

    X = pad_features(features, max_len=MAX_PAD_LENGTH)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Saved label encoder with classes: {label_encoder.classes_}")

    num_classes = len(label_encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioClassifier(N_MFCC, 128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save the model and check for errors
            try:
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info(f"Saved best model with val_acc: {best_val_acc:.4f}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("Training finished.")

if __name__ == '__main__':
    train()
