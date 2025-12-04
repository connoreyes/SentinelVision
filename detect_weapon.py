from ultralytics import YOLO
import cv2
class WeaponDetector:
    def __init__ (self, model_path = "models/best.pt"):
        self.model = YOLO(model_path)

    def detect_weapon(self, frame, conf_threshold=0.4):
        """
        Returns a list of bounding boxes:
        Each box = (x1, y1, x2, y2, confidence)
        """
        
        results = self.model(frame)

        boxes_weapon = []

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_weapon.append((x1, y1, x2, y2, conf))
        return boxes_weapon


