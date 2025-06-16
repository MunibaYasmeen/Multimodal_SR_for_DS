import librosa
from moviepy.editor import VideoFileClip

class AudioProcessor:
    def __init__(self):
        pass

    def extract_mfcc_from_video(self, video_path, sr=16000, n_mfcc=13):
        
        clip = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".wav")
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1).tolist()  # reduce temporal dim
