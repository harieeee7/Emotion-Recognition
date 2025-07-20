# Speech Emotion Recognition App 🎙️

This project is a real-time Speech Emotion Recognition system built using:

- 🧠 PyTorch (LSTM model)
- 🎧 MFCC feature extraction with Librosa
- 🎛️ Trained on the RAVDESS dataset
- 🌐 Deployed using Streamlit Cloud

### Features
- Record or upload audio
- Predict emotions like happy, sad, angry, etc.
- Visualize results with probability scores

### Deployment
Built to run directly on [Streamlit Cloud](https://streamlit.io/cloud). Just upload the model (`.pth`) and label encoder (`.pkl`) in the `models/` directory.
