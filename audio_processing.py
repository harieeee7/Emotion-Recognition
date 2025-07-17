import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Constants to match training config
SAMPLE_RATE = 16000
N_MFCC = 13
MAX_PAD_LENGTH = 0  # Will compute per inference based on trained max length

class AudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size//2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size//2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Last time step
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AudioProcessor:
    def __init__(self, model_path='models/audio_model.pth', max_pad_length=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        # Use the exact labels in the order matching training data 
        # (update if your training labels differ)
        self.label_encoder.fit(['neutral', 'happy', 'sad', 'angry', 'disgust', 'calm'])
        
        self.model = None
        self.model_path = model_path
        
        # Use max_pad_length if specified, otherwise default
        self.max_pad_length = max_pad_length or 100  # Default fallback, you might want to save max length during training
        
        self.load_model()

    def load_model(self):
        num_classes = len(self.label_encoder.classes_)
        input_size = N_MFCC
        hidden_size = 64
        
        self.model = AudioClassifier(input_size, hidden_size, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def extract_features(self, audio_path):
        try:
            if not os.path.exists(audio_path):
                return None, "File not found"

            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            if len(audio) < sr:  # minimum 1 second audio
                return None, "Audio too short (min 1 second required)"

            if librosa.feature.rms(y=audio).max() < 0.001:
                return None, "Silent audio detected"

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

            # Pad or trim according to max_pad_length set during initialization
            if mfccs.shape[1] > self.max_pad_length:
                mfccs = mfccs[:, :self.max_pad_length]
            else:
                pad_width = self.max_pad_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')

            return mfccs.T, None  # shape: (time, features)
        except Exception as e:
            return None, f"Audio processing error: {str(e)}"

    def predict_emotion(self, audio_path):
        features, error = self.extract_features(audio_path)
        if error:
            return {'error': error}
        try:
            features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probabilities)
                confidence = probabilities[pred_idx]
                label_names = list(self.label_encoder.classes_)
                all_predictions = dict(zip(label_names, probabilities.astype(float)))
            return {
                'emotion': label_names[pred_idx],
                'confidence': float(confidence),
                'all_predictions': all_predictions,
                'error': None
            }
        except Exception as e:
            return {'error': f"Prediction error: {str(e)}"}



def process_audio(audio_path):
    processor = AudioProcessor(model_path='models/audio_model.pth', max_pad_length=100)  # adjust max_pad_length to your training data max
    return processor.predict_emotion(audio_path)
