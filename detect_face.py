from ultralytics import YOLO   # Imports the YOLO object detection model from the Ultralytics package
import cv2                     # OpenCV used for image handling and cropping


# -------------------------------------------------------------------
# EXPAND A FACE BOUNDING BOX FOR MORE STABLE EMBEDDINGS
# -------------------------------------------------------------------
def expand_box(x1, y1, x2, y2, frame_shape, scale=0.3):
    """
    Slightly enlarges a detected face bounding box.

    WHY?
    - YOLO face boxes are often tight.
    - A larger crop helps DeepFace produce *more consistent embeddings*.
    """

    h, w, _ = frame_shape        # Extract the frame's height, width, and channels

    bw = x2 - x1                 # Width of the original bounding box
    bh = y2 - y1                 # Height of the original bounding box

    cx = x1 + bw // 2            # Center X of the original box
    cy = y1 + bh // 2            # Center Y of the original box

    # Increase box dimensions by `scale` (e.g., +30%)
    new_w = int(bw * (1 + scale))
    new_h = int(bh * (1 + scale))

    # Compute new box boundaries centered around the face
    nx1 = max(0, cx - new_w // 2)    # Ensure box does NOT go outside the frame
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(w, cx + new_w // 2)
    ny2 = min(h, cy + new_h // 2)

    return nx1, ny1, nx2, ny2


# -------------------------------------------------------------------
# FACE DETECTOR CLASS USING YOLOv8 FACE MODEL
# -------------------------------------------------------------------
class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt"):
        """
        Loads YOLO model for face detection.
        model_path: path to pretrained YOLO face detection weights.
        """
        self.model = YOLO(model_path)


    def detect_faces(self, frame):
        """
        Runs YOLO on the input frame and returns a list of detected faces.

        RETURN FORMAT:
            [
                (x1, y1, x2, y2, face_img)
            ]
        """

        # 1. Run inference using YOLO
        results = self.model(frame)

        faces = []  # Storage for detected faces

        # YOLO can return multiple detection batches; iterate through each
        for r in results:

            boxes = r.boxes               # YOLO bounding box output
            if boxes is None or len(boxes) == 0:
                continue                  # No faces detected — move to next image

            # ---------------------------------------------------------
            # SORT FACES BY SIZE — LARGEST FACE FIRST
            # ---------------------------------------------------------
            boxes = sorted(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) *    # width
                              (b.xyxy[0][3] - b.xyxy[0][1]),     # height
                reverse=True                                     # largest first
            )

            # Only keep the LARGEST detected face
            # WHY?
            # - Your facial ID system only tracks 1 identity at a time
            boxes = boxes[:1]

            # ---------------------------------------------------------
            # PROCESS THE REMAINING FACE BOX
            # ---------------------------------------------------------
            for box in boxes:

                # Extract YOLO bounding box coordinates and convert to int
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Expand the bounding box to include more context
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, frame.shape, scale=0.3)

                # Crop the detected face from the full frame
                face_img = frame[y1:y2, x1:x2].copy()

                # Safety check: if crop failed (box outside frame)
                if face_img.size == 0:
                    continue

                # Add face info to output list
                faces.append((x1, y1, x2, y2, face_img))

        # Return all detected faces (only 1 right now)
        return faces