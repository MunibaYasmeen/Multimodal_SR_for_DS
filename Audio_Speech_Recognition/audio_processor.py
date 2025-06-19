import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import os

class AudioProcessor:
    def __init__(self):
        pass

    def extract_mfcc_with_timestamps(self, video_path, sr=16000, n_mfcc=13, hop_length=512):
        print(f"🎞️ Processing: {video_path}")
        
        audio_path = video_path.replace(".mp4", ".wav")
        
        # Extract audio
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Audio file was not saved: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)
        print(f"🔊 Loaded audio: {len(y)} samples, Sample Rate: {sr}")

        if np.allclose(y, 0):
            raise ValueError("❌ Audio data is silent or empty.")

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        print(f"📈 MFCC shape: {mfcc.shape}")

        # Get timestamps
        times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)

        mfcc_with_timestamps = []
        for i, t in enumerate(times):
            vec = mfcc[:, i].tolist()
            if not np.allclose(vec, 0):  # Ignore silent/zero frames
                mfcc_with_timestamps.append({
                    "timestamp": round(t, 3),
                    "mfcc": vec
                })

        return mfcc_with_timestamps
