import os
import torch
import librosa
import numpy as np
import pickle
import streamlit as st
from io import BytesIO

# CONFIGURATION
MODEL_PATH = 'models/audio_model.pth'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LENGTH = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD ENCODER & MODEL
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

class AudioClassifier(torch.nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=8):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_size * 2, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = AudioClassifier(N_MFCC, 128, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# AUDIO HELPER FUNCTIONS
def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0) + 1e-9
    return (mfcc - mean) / std

def pad_features(mfcc, max_len=MAX_PAD_LENGTH):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    pad_width = max_len - mfcc.shape[0]
    return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

def preprocess_audio_file(audio_bytes):
    try:
        audio_np, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
        if len(audio_np) < sr:
            raise ValueError("Audio file is too short. Minimum 1 second required.")
        mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=N_MFCC)
        mfcc = normalize_mfcc(mfcc.T)
        mfcc = pad_features(mfcc)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise e

# STREAMLIT UI
st.set_page_config(page_title="Audio Emotion Classifier", layout="centered")
st.title("🗣️ Audio Emotion Recognition")
st.write("Upload a `.wav` audio file (at least 1 second long) to predict the emotion.")

uploaded_file = st.file_uploader("Upload your WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    audio_bytes_io = BytesIO(uploaded_file.getvalue())

    if st.button('Predict Emotion'):
        try:
            input_tensor = preprocess_audio_file(audio_bytes_io)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                emotion = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])

                st.success(f"🎉 Predicted Emotion: **{emotion}** with {confidence * 100:.2f}% confidence")

                st.subheader("📊 Prediction Probabilities")
                st.bar_chart({label: probs[i] for i, label in enumerate(label_encoder.classes_)})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
