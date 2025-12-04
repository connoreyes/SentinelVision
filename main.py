import cv2
from detect_face import FaceDetector
from detect_weapon import WeaponDetector
from utils.draw import draw_boxes, draw_weapon_boxes

def main():
    face_detector = FaceDetector()
    weapon_detector = WeaponDetector()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        face_boxes = face_detector.detect_face(frame)
        frame = draw_boxes(frame, face_boxes, label="Face")

        # Detect weapons
        weapon_boxes = weapon_detector.detect_weapon(frame)
        frame = draw_weapon_boxes(frame, weapon_boxes)

        cv2.imshow("SentinelVision", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()