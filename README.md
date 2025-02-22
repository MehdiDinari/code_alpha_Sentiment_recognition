# Vocal Emotion Detector

## Project Overview
The **Vocal Emotion Detector** is a Python-based application that records a short audio clip, extracts Mel-Frequency Cepstral Coefficients (MFCCs) features from the recording, and then predicts the emotion conveyed in the voice. The project is developed as part of my **internship at Code Alpha**.

## Features
- **Record Voice Input:** Captures a 3-second audio clip using the **sounddevice** library.
- **Feature Extraction:** Uses **librosa** to extract MFCC features.
- **Emotion Classification (Simulated):** Generates a random emotion prediction from a predefined set of emotions.
- **Graphical User Interface (GUI):** Built with **Tkinter** for user-friendly interaction.

## Technologies Used
- **Python 3.x**
- **Tkinter** (for GUI)
- **Librosa** (for audio processing)
- **Sounddevice** (for recording audio)
- **NumPy** (for numerical operations)
- **Random** (for emotion simulation)

## Installation & Setup
1. Clone the repository or download the source code.
2. Install the required dependencies:
   ```sh
   pip install numpy librosa sounddevice
   ```
3. Run the application:
   ```sh
   python main.py
   ```

## How It Works
1. Click on the **'Enregistrer'** button to start recording.
2. The application captures a 3-second audio clip.
3. MFCC features are extracted from the recorded audio.
4. A simulated prediction is made by selecting a random emotion.
5. The detected emotion is displayed on the GUI.

## Emotion Categories
The emotions recognized by the application (simulated) are:
- **Neutral**
- **Happy**
- **Sad**
- **Angry**
- **Fear**
- **Disgust**
- **Surprise**

## Future Enhancements
- Implement a machine learning model for real emotion detection.
- Improve feature extraction for better accuracy.
- Add support for longer audio recordings.
- Save and analyze past recordings.

## Author
Developed by **[Mehdi Dinari]** as part of an **internship with Code Alpha**.

## License
This project is open-source and available for educational and research purposes.

---
Feel free to modify this **README** file as needed to include additional details about the project or your internship experience!

