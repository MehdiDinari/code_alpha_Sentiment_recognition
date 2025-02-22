import streamlit as st
import numpy as np
import librosa
import librosa.feature
import random
import sounddevice as sd
import time

# Style personnalisé avec des emojis
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #4B0082;
        text-align: center;
    }
    .btn-record {
        background-color: #4B0082;
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 5px;
        font-size: 1.2rem;
        cursor: pointer;
    }
    .btn-record:hover {
        background-color: #5c2a9d;
    }
    .emoji {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True
)


def extract_features_from_audio(y, sr):
    """
    Extraction des caractéristiques MFCC avec dimension temporelle à partir d'un signal audio 🎵
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Ajustement pour obtenir 130 trames (pour un audio de 3s)
        if mfcc.shape[1] < 130:
            pad_width = 130 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :130]
        # Reshape pour ajouter la dimension de canal (batch, n_mfcc, temps, canal)
        mfcc = mfcc.reshape(1, 40, 130, 1)
        return mfcc
    except Exception as e:
        st.error("⚠️ Erreur lors de l'extraction des caractéristiques : " + str(e))
        return None


def get_emotion_label(class_idx):
    """
    Associe l'indice de la classe aux labels d'émotion 🙂
    """
    emotions = ['neutral 😐', 'happy 😊', 'sad 😢', 'angry 😠', 'fear 😱', 'disgust 🤢', 'surprise 😮']
    return emotions[class_idx] if class_idx < len(emotions) else "unknown"


def record_audio(duration=3, sr=22050):
    st.info("🎤 Enregistrement en cours...")
    time.sleep(0.5)  # Petite pause pour afficher le message
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # Attendre la fin de l'enregistrement
        audio = audio.flatten()  # Conversion en tableau 1D
        return audio, sr
    except Exception as e:
        st.error("⚠️ Erreur lors de l'enregistrement : " + str(e))
        return None, None


def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.title("Vocal Emotion Detector 🎙️")
    st.write("Enregistrez un audio de 3 secondes pour détecter l'émotion. 😃😢😡")

    if st.button("Enregistrer 🎤", key="record", help="Cliquez pour enregistrer 3 secondes d'audio"):
        with st.spinner("Enregistrement et traitement en cours... ⏳"):
            audio, sr = record_audio()
            if audio is not None:
                features = extract_features_from_audio(audio, sr)
                if features is not None:
                    # Simulation d'une prédiction aléatoire
                    predicted_index = random.randint(0, 6)
                    emotion = get_emotion_label(predicted_index)
                    st.success(f"Émotion détectée : **{emotion}**")
                else:
                    st.error("⚠️ Erreur lors du traitement de l'enregistrement.")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
