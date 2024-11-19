import librosa
import soundfile as sf
import wave
import os
import warnings
warnings.filterwarnings('ignore')

for (root, dirs, files) in os.walk('/data_old/train/dr1', topdown=True):
    if len(dirs) == 0:
        for file in files:
            if file.endswith(".wav"):
                print(f"Converting file {file} to sox")

                x, _ = librosa.load(file, sr=16000)
                sf.write(file, x, 16000)
                # wave.open(file, 'r')


