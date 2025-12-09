# Serial communication with Arduino + time delays if needed
import serial, time

# OpenCV for camera frames and drawing
import cv2

# Numerical processing (embedding math)
import numpy as np

# YOLOv8 model loader from Ultralytics
from ultralytics import YOLO

# Python threading → lets webcam run at high FPS independently
from threading import Thread

# Used to timestamp identity sightings
from datetime import datetime

# Our custom YOLO-based face detector (bigger crop box etc.)
from detect_face import FaceDetector

# Local SQLite database for identities + sightings
from database import FaceDatabase

# Identity engine (ArcFace embeddings + threshold matching)
from identify_person import FaceIdentifier


# ------------------------------------------------------------
# CONNECT TO ARDUINO
# ------------------------------------------------------------
# Opens serial port at 115200 baud.
# Replace with the correct port for your system.
ser = serial.Serial('/dev/cu.usbmodem2101', 115200)


# =====================================================================
# HIGH-FPS WEBCAM THREAD
# =====================================================================
class WebcamStream:
    def __init__(self, src=0):
        # Start video capture on default camera index 0
        self.cap = cv2.VideoCapture(src)

        # Read one initial frame
        self.ret, self.frame = self.cap.read()

        # Flag to keep update() running
        self.running = True

        # Start background thread that constantly pulls new frames
        Thread(target=self.update, daemon=True).start()

    def update(self):
        # Runs forever until self.running = False
        while self.running:
            # Always grab the newest frame → high FPS effect
            self.ret, self.frame = self.cap.read()

    def read(self):
        # Returns the most recently captured frame
        return self.frame

    def stop(self):
        # Stop reading and release hardware
        self.running = False
        self.cap.release()


# =====================================================================
# LOAD MODELS + DATABASE
# =====================================================================
print("[INFO] Loading models...")

# YOLOv8 face detector (lightweight, small model)
face_model = YOLO("models/yolov8n-face.pt")

# YOLOv8 weapon detector (my rained model)
weapon_model = YOLO("models/yolov8s-weapon.pt")

# Initialize the local SQLite database
db = FaceDatabase()

# Identity engine (ArcFace + EMA refinement)
identifier = FaceIdentifier(
    db=db,
    threshold=0.40,   # max L2 distance for match
    alpha=0.12        # how strongly new embeddings influence canonical one
)

print(f"[INFO] Loaded {db.count_persons()} known persons.")

# Start the high-FPS camera thread
cap = WebcamStream(0)

# YOLO face detector class (your improved version with box expansion)
face_detector = FaceDetector("models/yolov8n-face.pt")

print("[INFO] SentinelVision running... Press 'q' to quit.")


# =====================================================================
# MAIN FRAME PROCESSING LOOP
# =====================================================================
while True:

    frame = cap.read()        # Grab latest frame from webcam thread
    if frame is None:
        continue              # If no frame, skip to next loop

    # Current timestamp → used in identity history logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # ========================================================
    # 1. FACE DETECTION (YOLOv8)
    # ========================================================
    face_results = face_model.predict(frame, imgsz=320, verbose=False)
    detected_faces = []       # Will store bounding boxes + identity IDs

    for r in face_results:
        for box in r.boxes:

            # Extract box coordinates as ints
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop out the face region from the frame
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # ---------------------------------------------------------
            # ARC-FACE EMBEDDING + IDENTITY MATCHING
            # ---------------------------------------------------------
            try:
                embedding = identifier.get_embedding(face_img)
            except Exception:
                # ArcFace occasionally fails → skip that frame
                continue

            # Determine if new or existing person
            person_id, dist, is_new = identifier.identify_or_create(
                embedding,
                timestamp=timestamp
            )

            # Log ID output
            if is_new:
                print(f"[NEW] Person {person_id} detected (dist={dist:.4f})")
            else:
                print(f"[MATCH] Person {person_id} (dist={dist:.4f})")

            # Save detected face w/ identity
            detected_faces.append((x1, y1, x2, y2, person_id))


    # ========================================================
    # 2. WEAPON DETECTION
    # ========================================================
    weapon_results = weapon_model.predict(frame, imgsz=320, verbose=False)
    weapon_detected = False

    for r in weapon_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.45:
                continue

            weapon_detected = True

            # Draw weapon detection box
            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Weapon {conf:.2f}",
                (wx1, wy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


    # ========================================================
    # 3. THREAT LEVEL COMPUTATION + ARDUINO OUTPUT
    # ========================================================
    # Detect at least one face
    faces = face_detector.detect_faces(frame)

    if len(faces) > 0:
        person = 1  # Person present
        x1, y1, x2, y2, face_img = faces[0]

        # Compute face center for turret tracking
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
    else:
        person = 0  # No detected face
        cx, cy = 0, 0

    # Determine overall threat level
    if weapon_detected:
        threat = 90
        status = "HIGH THREAT"
        color = (0, 0, 255)
    else:
        # Harmless person
        if person == 1:
            threat = 15
        else:
            threat = 0
        status = "LOW THREAT"
        color = (0, 255, 0)

    # --------------------------------------------------------
    # SEND DATA TO ARDUINO
    # --------------------------------------------------------
    # Arduino expects consistent formatting:
    # PERSON:1,THREAT:90,X:350,Y:200
    msg = f"PERSON:{person},THREAT:{threat},X:{cx},Y:{cy}\n"
    ser.write(msg.encode())


    # Draw threat text onscreen
    cv2.putText(frame, f"Threat: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


    # ========================================================
    # 4. DRAW FACE BOXES + PERSON IDs
    # ========================================================
    for (x1, y1, x2, y2, pid) in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"Person {pid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


    # ========================================================
    # 5. SHOW LIVE DISPLAY
    # ========================================================
    cv2.imshow("SentinelVision — Identity Engine v2.0", frame)

    # Quit on "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# SHUTDOWN
cap.stop()
cv2.destroyAllWindows()