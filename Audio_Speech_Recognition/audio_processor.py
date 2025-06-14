import subprocess
import librosa
import ffmpeg
 
class AudioProcessor:
    def __init__(self):
        pass

    def convert_video_to_audio(self, input_file, output_file):
        ffmpeg_cmd = {
            "ffmpeg",
            "-i", input_file,
            "-vn",
            "-acodec","libmp3lame",
            "-ab", "192k",
            "-ar", "44100",
            "-y",
            output_file
        }

        try:
            subprocess.run(ffmpeg_cmd, check = True)
            print("Successfully Converted!")
        except subprocess.CalledProcessError as e:
            print("Conversion Failed!")

    def extract_mfcc_from_video(self, audio_file , sr=16000, n_mfcc=13):
        y, sr = librosa.load(audio_file, sr=sr) 
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1).tolist()  # reduce temporal dim
