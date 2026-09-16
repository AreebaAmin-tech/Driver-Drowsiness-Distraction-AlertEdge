import cv2
import numpy as np

# Canonical 3D facial model points (in millimeters)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye left corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye right corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float64)

KEY_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

class HeadPoseEstimator:
    def __init__(self, yaw_thresh=25.0, pitch_thresh=20.0, consecutive_frames=15):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.consecutive_frames = consecutive_frames
        self.distracted_frame_count = 0

    def get_pose(self, mesh_points, frame_w, frame_h):
        face_2d = []
        for idx in KEY_LANDMARK_IDS:
            pt = mesh_points[idx]
            face_2d.append([pt.x * frame_w, pt.y * frame_h])
        face_2d = np.array(face_2d, dtype=np.float64)

        # Approximate camera intrinsics
        focal_length = frame_w
        cam_matrix = np.array([
            [focal_length, 0, frame_w / 2.0],
            [0, focal_length, frame_h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return False, 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # RQ decomposition returns angles in degrees directly
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize pitch to [-90, 90] range (handles camera axis flip)
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        # Persistence check: must look away for consecutive frames
        is_deviated = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh
        is_distracted = False

        if is_deviated:
            self.distracted_frame_count += 1
            if self.distracted_frame_count >= self.consecutive_frames:
                is_distracted = True
        else:
            self.distracted_frame_count = 0

        return is_distracted, pitch, yaw, roll