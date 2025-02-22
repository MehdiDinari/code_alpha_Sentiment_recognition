import tkinter as tk
from tkinter import ttk
import numpy as np
import librosa
import librosa.feature
import random  # pour simuler une prédiction
import sounddevice as sd  # pour l'enregistrement audio

class EmotionDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Vocal Emotion Detector")
        master.geometry("400x300")
        master.configure(bg='#f5f5f5')

        # Configuration du style avec ttk
        style = ttk.Style(master)
        style.theme_use('clam')
        style.configure('TLabel', font=('Helvetica', 12), background='#f5f5f5')
        style.configure('TButton', font=('Helvetica', 12), padding=6)

        # Création d'un cadre principal
        self.frame = ttk.Frame(master, padding="20 20 20 20")
        self.frame.pack(expand=True)

        # Titre de l'application
        self.title_label = ttk.Label(self.frame, text="Vocal Emotion Detector", font=('Helvetica', 16, 'bold'))
        self.title_label.pack(pady=(0, 15))

        # Instruction pour l'utilisateur
        self.instruction_label = ttk.Label(self.frame, text="Enregistrez un audio (3 secondes)")
        self.instruction_label.pack(pady=(0, 10))

        # Bouton d'enregistrement
        self.button = ttk.Button(self.frame, text="Enregistrer", command=self.record_audio)
        self.button.pack(pady=10)

        # Label pour afficher le résultat
        self.result_label = ttk.Label(self.frame, text="", font=('Helvetica', 12, 'italic'))
        self.result_label.pack(pady=(15, 10))

    def extract_features_from_audio(self, y, sr):
        """Extraction des caractéristiques MFCC avec dimension temporelle à partir d'un signal audio"""
        try:
            # Extraction des MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

            # Ajustement pour obtenir 130 trames (pour un audio de 3s)
            if mfcc.shape[1] < 130:
                pad_width = 130 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :130]

            # Reshape pour ajouter la dimension de canal
            mfcc = mfcc.reshape(1, 40, 130, 1)  # (batch, n_mfcc, temps, canal)
            return mfcc
        except Exception as e:
            print("Erreur lors de l'extraction des caractéristiques :", e)
            return None

    def record_audio(self):
        duration = 3      # durée de l'enregistrement en secondes
        sr = 22050        # fréquence d'échantillonnage
        self.result_label.config(text="Enregistrement en cours...")
        self.master.update()  # mise à jour de l'interface

        try:
            # Enregistrement audio
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
            sd.wait()  # attendre la fin de l'enregistrement
            audio = audio.flatten()  # convertir en tableau 1D
        except Exception as e:
            print("Erreur lors de l'enregistrement :", e)
            self.result_label.config(text="Erreur lors de l'enregistrement")
            return

        features = self.extract_features_from_audio(audio, sr)
        if features is not None:
            # Simulation d'une prédiction
            predicted_index = random.randint(0, 6)
            emotion = self.get_emotion_label(predicted_index)
            self.result_label.config(text=f"Émotion détectée : {emotion}")
        else:
            self.result_label.config(text="Erreur lors du traitement de l'enregistrement")

    def get_emotion_label(self, class_idx):
        """Associe l'indice de la classe aux labels d'émotion"""
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        return emotions[class_idx] if class_idx < len(emotions) else "unknown"

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
