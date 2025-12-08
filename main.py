# main.py — FULL STABLE IDENTITY VERSION (Option B)
import serial, time
import cv2
import numpy as np
from ultralytics import YOLO
from threading import Thread
from datetime import datetime
from detect_face import FaceDetector

from database import FaceDatabase
from identify_person import FaceIdentifier   # ← THIS IS YOUR NEW ID ENGINE

ser = serial.Serial('/dev/cu.usbmodem2101', 115200)
# ============================================================
# HIGH-FPS WEBCAM THREAD
# ============================================================
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# ============================================================
# LOAD MODELS + DATABASE
# ============================================================
print("[INFO] Loading models...")

face_model = YOLO("models/yolov8n-face.pt")
weapon_model = YOLO("models/yolov8s-weapon.pt")

db = FaceDatabase()
identifier = FaceIdentifier(db=db, threshold=0.40, alpha=0.12)

print(f"[INFO] Loaded {db.count_persons()} known persons.")

# webcam thread
cap = WebcamStream(0)
# Face detector
face_detector = FaceDetector("models/yolov8n-face.pt")

print("[INFO] SentinelVision running... Press 'q' to quit.")


# ============================================================
# FRAME LOOP
# ============================================================
while True:

    frame = cap.read()
    if frame is None:
        continue

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --------------------------------------------------------
    # 1. FACE DETECTION
    # --------------------------------------------------------
    face_results = face_model.predict(frame, imgsz=320, verbose=False)
    detected_faces = []

    for r in face_results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # -----------------------------------------
            # ARC-FACE EMBEDDING → MATCH OR NEW PERSON
            # -----------------------------------------
            try:
                embedding = identifier.get_embedding(face_img)
            except:
                continue

            person_id, dist, is_new = identifier.identify_or_create(
                embedding, timestamp=timestamp
            )

            if is_new:
                print(f"[NEW] Person {person_id} detected (dist={dist:.4f})")
            else:
                print(f"[MATCH] Person {person_id} (dist={dist:.4f})")

            detected_faces.append((x1, y1, x2, y2, person_id))


    # --------------------------------------------------------
    # 2. WEAPON DETECTION
    # --------------------------------------------------------
    weapon_results = weapon_model.predict(frame, imgsz=320, verbose=False)
    weapon_detected = False

    for r in weapon_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.45:
                continue

            weapon_detected = True
            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
            cv2.putText(frame, f"Weapon {conf:.2f}", (wx1, wy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # --------------------------------------------------------
    # 3. THREAT LEVEL DISPLAY
    # --------------------------------------------------------
    # arduino threat detection
    # default location if no faces
    faces = face_detector.detect_faces(frame)

    if len(faces) > 0:
        person = 1
        x1, y1, x2, y2, face_img = faces[0]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
    else:
        person = 0
        cx, cy = 0, 0

    if weapon_detected:
        threat = 90
        status = "HIGH THREAT"
        color = (0, 0, 255)
    else:
        if person == 1:
            threat = 15
        else:
            threat = 0
        status = "LOW THREAT"
        color = (0, 255, 0)

    # arduino 
    # Format message EXACTLY how Arduino expects
    msg = f"PERSON:{person},THREAT:{threat},X:{cx},Y:{cy}\n"

    ser.write(msg.encode())


    cv2.putText(frame, f"Threat: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


    # --------------------------------------------------------
    # 4. DRAW FACE BOXES + IDs
    # --------------------------------------------------------
    for (x1, y1, x2, y2, pid) in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


    # --------------------------------------------------------
    # 5. SHOW FRAME
    # --------------------------------------------------------
    cv2.imshow("SentinelVision — Identity Engine v2.0", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cleanup
cap.stop()
cv2.destroyAllWindows()