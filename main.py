import time
import cv2
import mediapipe as mp
import numpy as np

from core.eye_detector import EyeDetector
from core.mouth_detector import MouthDetector
from core.head_pose import HeadPoseEstimator
from core.object_detector import EdgeObjectDetector
from core.fatigue_scorer import FatigueScorer
from utils.sound_alert import AlertSound


def draw_modern_hud(frame, metrics, alerts, fps):
    h, w, _ = frame.shape

    # Proportional scaling factors relative to standard 720p base
    scale = max(0.55, min(w / 1280.0, 1.0))
    font = cv2.FONT_HERSHEY_DUPLEX

    # --- 1. TOP ALERT PILL ---
    is_alert = len(alerts) > 0
    pill_color = (35, 35, 215) if is_alert else (40, 160, 60)
    pill_text = " | ".join(alerts) if is_alert else "STATUS: ATTENTIVE"

    f_scale_pill = 0.65 * scale
    thick_pill = max(1, int(2 * scale))
    (tw, th), _ = cv2.getTextSize(pill_text, font, f_scale_pill, thick_pill)

    pill_w = tw + int(30 * scale)
    pill_h = int(38 * scale)
    p_x1, p_y1 = int(16 * scale), int(14 * scale)
    p_x2, p_y2 = p_x1 + pill_w, p_y1 + pill_h

    sub = frame[p_y1:p_y2, p_x1:p_x2]
    rect = np.full_like(sub, pill_color)
    cv2.addWeighted(rect, 0.88, sub, 0.12, 0, sub)
    frame[p_y1:p_y2, p_x1:p_x2] = sub
    cv2.putText(frame, pill_text, (p_x1 + int(14 * scale), p_y1 + int(26 * scale)),
                font, f_scale_pill, (255, 255, 255), thick_pill, cv2.LINE_AA)

    # Top-Right FPS Pill
    fps_text = f"{fps:.1f} FPS"
    f_scale_fps = 0.60 * scale
    (ftw, fth), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, f_scale_fps, thick_pill)
    fps_w = ftw + int(24 * scale)
    fps_x1 = w - fps_w - int(16 * scale)
    fps_sub = frame[p_y1:p_y2, fps_x1:fps_x1 + fps_w]
    fps_bg = np.full_like(fps_sub, (20, 24, 28))
    cv2.addWeighted(fps_bg, 0.80, fps_sub, 0.20, 0, fps_sub)
    frame[p_y1:p_y2, fps_x1:fps_x1 + fps_w] = fps_sub
    cv2.putText(frame, fps_text, (fps_x1 + int(12 * scale), p_y1 + int(26 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_scale_fps, (200, 240, 200), thick_pill, cv2.LINE_AA)

    # --- 2. BOTTOM DOCK ---
    dock_h = int(82 * scale)
    dock_y1 = h - dock_h
    dock_sub = frame[dock_y1:h, 0:w]
    dock_bg = np.full_like(dock_sub, (16, 18, 22))
    cv2.addWeighted(dock_bg, 0.85, dock_sub, 0.15, 0, dock_sub)
    frame[dock_y1:h, 0:w] = dock_sub
    cv2.line(frame, (0, dock_y1), (w, dock_y1), (50, 55, 65), max(1, int(1.5 * scale)))

    col_w = w // 4
    f_header = 0.50 * scale
    f_body = 0.60 * scale
    f_sub = 0.46 * scale
    thick_main = max(1, int(1.8 * scale))
    thick_sub = 1

    # Col 1: Eyes
    ear_color = (50, 50, 245) if metrics["ear"] < 0.20 else (240, 240, 240)
    cv2.putText(frame, "EYES", (int(20 * scale), dock_y1 + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_header, (0, 215, 255), thick_sub, cv2.LINE_AA)
    cv2.putText(frame, f"EAR: {metrics['ear']:.2f}", (int(20 * scale), dock_y1 + int(48 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_body, ear_color, thick_main, cv2.LINE_AA)
    cv2.putText(frame, f"PERCLOS: {metrics['perclos']:.0f}%", (int(20 * scale), dock_y1 + int(70 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_sub, (180, 180, 180), thick_sub, cv2.LINE_AA)

    # Col 2: Mouth
    mar_color = (0, 180, 255) if metrics["mar"] > 0.65 else (240, 240, 240)
    cv2.putText(frame, "MOUTH", (col_w + int(10 * scale), dock_y1 + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_header, (0, 215, 255), thick_sub, cv2.LINE_AA)
    cv2.putText(frame, f"MAR: {metrics['mar']:.2f}", (col_w + int(10 * scale), dock_y1 + int(48 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_body, mar_color, thick_main, cv2.LINE_AA)
    cv2.putText(frame, f"Yawns: {metrics['yawns']}", (col_w + int(10 * scale), dock_y1 + int(70 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_sub, (180, 180, 180), thick_sub, cv2.LINE_AA)

    # Col 3: Pose
    pose_alert = abs(metrics["yaw"]) > 25.0 or abs(metrics["pitch"]) > 20.0
    pose_color = (50, 50, 245) if pose_alert else (240, 240, 240)
    cv2.putText(frame, "HEAD POSE", (col_w * 2 + int(10 * scale), dock_y1 + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_header, (0, 215, 255), thick_sub, cv2.LINE_AA)
    cv2.putText(frame, f"Yaw: {metrics['yaw']:.1f}", (col_w * 2 + int(10 * scale), dock_y1 + int(48 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_body, pose_color, thick_main, cv2.LINE_AA)
    cv2.putText(frame, f"Pitch: {metrics['pitch']:.1f}", (col_w * 2 + int(10 * scale), dock_y1 + int(70 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_sub, (180, 180, 180), thick_sub, cv2.LINE_AA)

    # Col 4: Fatigue Bar
    fatigue = metrics["fatigue_idx"]
    f_color = (50, 220, 50) if fatigue < 40 else ((0, 180, 255) if fatigue < 60 else (40, 40, 255))
    cv2.putText(frame, f"FATIGUE: {fatigue:.0f}%", (col_w * 3 + int(10 * scale), dock_y1 + int(24 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, f_body, f_color, thick_main, cv2.LINE_AA)

    bar_x = col_w * 3 + int(10 * scale)
    bar_w = col_w - int(40 * scale)
    bar_h = int(16 * scale)
    bar_y = dock_y1 + int(40 * scale)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 44, 52), -1)
    fill_w = int(bar_x + (fatigue / 100.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (fill_w, bar_y + bar_h), f_color, -1)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Use native widescreen capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Edge Driver Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set standard initial 16:9 window size
    cv2.resizeWindow(window_name, 960, 540)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    eye_detector = EyeDetector(ear_threshold=0.20, consecutive_frames=9)
    mouth_detector = MouthDetector(mar_threshold=0.65, consecutive_frames=12)
    pose_estimator = HeadPoseEstimator()
    object_detector = EdgeObjectDetector(model_path="models/yolov8n.onnx", conf_thresh=0.50)
    fatigue_scorer = FatigueScorer(window_seconds=60)
    alert = AlertSound()

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    phone_hold_counter = 0
    phone_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time

        if frame_count % 2 == 0:
            detected, raw_boxes = object_detector.detect_distraction(frame)
            if detected:
                phone_hold_counter = 5
                phone_boxes = raw_boxes
            else:
                if phone_hold_counter > 0:
                    phone_hold_counter -= 1
                else:
                    phone_boxes = []

        phone_detected = phone_hold_counter > 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_alerts = []

        if phone_detected:
            status_alerts.append("PHONE USE")
            alert.trigger()
            for (bx1, by1, bx2, by2, score) in phone_boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 255), 2)
                cv2.putText(frame, f"Phone {score:.2f}", (bx1, max(25, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 255), 2, cv2.LINE_AA)

        metrics = {
            "ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
            "fatigue_idx": 0.0, "yawns": 0, "perclos": 0.0
        }

        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark

            is_drowsy, left_ear, right_ear, avg_ear = eye_detector.evaluate(mesh_points, w, h)
            is_yawn, mar = mouth_detector.evaluate(mesh_points, w, h)
            is_pose_distracted, pitch, yaw, roll = pose_estimator.get_pose(mesh_points, w, h)

            instant_eyes_closed = avg_ear < eye_detector.ear_threshold
            instant_distracted = is_pose_distracted or phone_detected
            fatigue_idx, yawns_60s, perclos_val = fatigue_scorer.update(
                is_eyes_closed=instant_eyes_closed,
                is_yawning=is_yawn,
                is_distracted=instant_distracted
            )

            metrics.update({
                "ear": avg_ear, "mar": mar, "pitch": pitch,
                "yaw": yaw, "fatigue_idx": fatigue_idx,
                "yawns": yawns_60s, "perclos": perclos_val
            })

            if is_drowsy:
                status_alerts.append("DROWSINESS DETECTED")
                alert.trigger()
            if is_yawn:
                status_alerts.append("YAWNING")
                alert.trigger()
            if is_pose_distracted:
                status_alerts.append("DISTRACTED GAZE")
                alert.trigger()
            if fatigue_idx >= 60.0:
                status_alerts.append("FATIGUE WARNING")
                alert.trigger()

        # Render dynamically scaled HUD
        out_frame = draw_modern_hud(frame, metrics, status_alerts, fps)

        # Detect active window size to avoid blurry nearest-neighbor OS blitting
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 100 and win_h > 100 and (win_w != w or win_h != h):
                # High quality area downsampling prevents pixel smearing
                display_frame = cv2.resize(out_frame, (win_w, win_h), interpolation=cv2.INTER_AREA)
            else:
                display_frame = out_frame
        except Exception:
            display_frame = out_frame

        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()