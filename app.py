import streamlit as st
import torch
import numpy as np
import librosa
from io import BytesIO
import pickle
import os

# Config
MODEL_PATH = 'models/audio_model.pth'
ENCODER_PATH = 'models/label_encoder.pkl'
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LENGTH = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

# Define model
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

# Load model
model = AudioClassifier(N_MFCC, 128, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Streamlit UI
st.set_page_config("Speech Emotion Recognition")
st.title("🎙️ Speech Emotion Detection")
st.write("Upload a `.wav` file to analyze emotions.")

uploaded_file = st.file_uploader("Upload your WAV file", type=["wav"])

def normalize_mfcc(mfcc):
    return (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-9)

def pad_features(mfcc, max_len=MAX_PAD_LENGTH):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    return np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')

def preprocess(file_bytes):
    y, sr = librosa.load(file_bytes, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    norm = normalize_mfcc(mfcc.T)
    padded = pad_features(norm)
    return torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Emotion"):
        try:
            input_tensor = preprocess(BytesIO(uploaded_file.read()))
            with torch.no_grad():
                out = model(input_tensor)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(prob))
                emotion = label_encoder.inverse_transform([pred_idx])[0]

            st.success(f"Predicted Emotion: **{emotion}** ({prob[pred_idx]*100:.2f}%)")
            st.bar_chart({label_encoder.classes_[i]: float(prob[i]) for i in range(len(prob))})

        except Exception as e:
            st.error(f"Prediction failed: {e}")
