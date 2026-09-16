import numpy as np

# Landmark indices for mouth aperture
MOUTH_INDICES = [61, 291, 0, 17, 84, 314]

def calculate_mar(mouth_landmarks):
    # Vertical distances
    dist_v1 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
    dist_v2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[5])
    
    # Horizontal distance
    dist_h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    
    if dist_h == 0:
        return 0.0
        
    return (dist_v1 + dist_v2) / (2.0 * dist_h)

class MouthDetector:
    def __init__(self, mar_threshold=0.65, consecutive_frames=15):
        self.mar_threshold = mar_threshold
        self.consecutive_frames = consecutive_frames
        self.yawn_frame_count = 0

    def evaluate(self, mesh_points, frame_w, frame_h):
        mouth_pts = np.array([[mesh_points[i].x * frame_w, mesh_points[i].y * frame_h] for i in MOUTH_INDICES])
        mar = calculate_mar(mouth_pts)

        is_yawning = False
        if mar > self.mar_threshold:
            self.yawn_frame_count += 1
            if self.yawn_frame_count >= self.consecutive_frames:
                is_yawning = True
        else:
            self.yawn_frame_count = 0

        return is_yawning, mar