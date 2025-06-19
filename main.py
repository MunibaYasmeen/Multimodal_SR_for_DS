from Audio_Speech_Recognition.audio_processor import AudioProcessor
from Video_Speech_Recognition.video_processing import extract_lip_landmarks
from pymongo import MongoClient
import os

# Initialize processors
audio_processor = AudioProcessor()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["speech_ds"]
collection = db["features"]

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
    
                audio_feat = audio_processor.extract_mfcc_with_timestamps(video_path)
                visual_feat = extract_lip_landmarks(video_path)

                # Save to MongoDB
                doc = {
                    "filename": file,
                    "label": label,
                    "audio_features": audio_feat,  
                    "visual_features": visual_feat,
                    "modality": "multimodal"
                }

                collection.insert_one(doc)
                print(f"✅ Inserted: {file}")
            
            except Exception as e:
                print(f"❌ Failed on {file}: {e}")
