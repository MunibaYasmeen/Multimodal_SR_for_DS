from Audio_Speech_Recognition.audio_processor import AudioProcessor
from Video_Speech_Recognition.video_processing import extract_lip_landmarks
from pymongo import MongoClient
import os

# Initialize processors
audio_processor = AudioProcessor()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["speech_ds"]
metadata_col = db["features"]
audio_col = db["audio_features"]
visual_col = db["visual_features"]

# Dataset folders
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
classes = {
    "normal_people": 0,
    "down_syndrome_people": 1
}

# Loop through each class folder
for class_name, label in classes.items():
    folder_path = os.path.join(base_path, class_name)

    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(folder_path, file)
            
            try:
                print(f"🧠 Processing {file}...")

                # Extract features
                audio_feat = audio_processor.extract_mfcc_with_timestamps(video_path)
                visual_feat = extract_lip_landmarks(video_path)

                # Save metadata
                metadata_col.insert_one({
                    "filename": file,
                    "label": label,
                    "modality": "multimodal"
                })

                # Save audio features
                audio_col.insert_one({
                    "filename": file,
                    "audio_features": audio_feat
                })

                # Save visual features
                visual_col.insert_one({
                    "filename": file,
                    "visual_features": visual_feat
                })

                print(f"✅ Inserted: {file}")

            except Exception as e:
                print(f"❌ Failed on {file}: {e}")
