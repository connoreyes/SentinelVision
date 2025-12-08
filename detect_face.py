from ultralytics import YOLO
import cv2




def expand_box(x1, y1, x2, y2, frame_shape, scale=0.3):
    h, w, _ = frame_shape
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    new_w = int(bw * (1 + scale))
    new_h = int(bh * (1 + scale))

    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(w, cx + new_w // 2)
    ny2 = min(h, cy + new_h // 2)

    return nx1, ny1, nx2, ny2

class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt"):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        """
        Returns: [(x1, y1, x2, y2, face_img)]
        """
        results = self.model(frame)
        faces = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # sort by area (largest first)
            boxes = sorted(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) *
                              (b.xyxy[0][3] - b.xyxy[0][1]),
                reverse=True
            )

            # keep only the largest face for now
            boxes = boxes[:1]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # expand box a bit for more consistent embeddings
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, frame.shape, scale=0.3)

                face_img = frame[y1:y2, x1:x2].copy()
                if face_img.size == 0:
                    continue


                faces.append((x1, y1, x2, y2, face_img))


        return faces