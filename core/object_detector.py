import cv2
import numpy as np
import onnxruntime as ort

class EdgeObjectDetector:
    def __init__(self, model_path="models/yolov8n.onnx", conf_thresh=0.50):
        self.conf_thresh = conf_thresh
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.target_class_id = 67  # COCO class 67 is cell phone

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(640 / h, 640 / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        top = (640 - nh) // 2
        left = (640 - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, left, top

    def detect_distraction(self, frame):
        tensor, scale, left, top = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})[0]
        predictions = np.squeeze(outputs).T

        phone_detected = False
        boxes = []

        for pred in predictions:
            class_scores = pred[4:]
            max_score = float(np.max(class_scores))
            class_id = int(np.argmax(class_scores))

            # Strictly require conf >= 0.50 to avoid hand false triggers
            if class_id == self.target_class_id and max_score >= self.conf_thresh:
                xc, yc, bw, bh = pred[:4]
                x1 = int((xc - bw / 2 - left) / scale)
                y1 = int((yc - bh / 2 - top) / scale)
                x2 = int((xc + bw / 2 - left) / scale)
                y2 = int((yc + bh / 2 - top) / scale)

                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                aspect = box_h / float(box_w)

                # Phones held upright or sideways have distinct aspect ratios and minimum size
                if (aspect > 1.2 or aspect < 0.8) and (box_w > 45 and box_h > 45):
                    phone_detected = True
                    boxes.append((x1, y1, x2, y2, max_score))

        return phone_detected, boxes