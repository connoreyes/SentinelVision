from  ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, mode_path = "models/yolov8n-face.pt"):
        # create instance of the model
        self.model = YOLO(mode_path)

    def detect_face(self, frame):
        """
        Returns a list of bounding boxes:
        Each box = (x1, y1, x2, y2)
        """

        # result is set to the frame 
        results = self.model(frame)
        # create a list boxes
        boxes = []

        # loop through each frame result
        for r in results:
            # loop through each box detection
            for box in r.boxes:
                # maps an integer and coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # append to boxes
                boxes.append((x1, y1, x2, y2))
        return boxes