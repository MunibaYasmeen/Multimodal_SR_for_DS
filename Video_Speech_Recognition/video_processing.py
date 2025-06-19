import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LIP_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
]

def extract_lip_landmarks(video_path):
    """Extract 2D lip landmarks from video using MediaPipe."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    visual_features = []

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            timestamp = frame_num / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                lips = [{"id": i, "x": landmarks[i].x, "y": landmarks[i].y} for i in LIP_IDX]
                visual_features.append({
                    "timestamp": timestamp,
                    "landmarks": lips
                })

            frame_num += 1

    cap.release()
    return visual_features
