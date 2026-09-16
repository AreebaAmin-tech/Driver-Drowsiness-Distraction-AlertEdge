import numpy as np

# Landmark indices for MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    # Vertical distances between upper and lower eyelids
    dist_v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    dist_v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance between eye corners
    dist_h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if dist_h == 0:
        return 0.0
    return (dist_v1 + dist_v2) / (2.0 * dist_h)

class EyeDetector:
    def __init__(self, ear_threshold=0.22, consecutive_frames=18):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.drowsy_frame_count = 0

    def evaluate(self, mesh_points, frame_w, frame_h):
        left_eye = np.array([[mesh_points[i].x * frame_w, mesh_points[i].y * frame_h] for i in LEFT_EYE_INDICES])
        right_eye = np.array([[mesh_points[i].x * frame_w, mesh_points[i].y * frame_h] for i in RIGHT_EYE_INDICES])
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        is_drowsy = False
        if avg_ear < self.ear_threshold:
            self.drowsy_frame_count += 1
            if self.drowsy_frame_count >= self.consecutive_frames:
                is_drowsy = True
        else:
            self.drowsy_frame_count = 0

        return is_drowsy, left_ear, right_ear, avg_ear