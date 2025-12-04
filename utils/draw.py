import cv2

def draw_boxes(frame, boxes, color=(0, 0, 255), label="Face"):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Text to display
        text = f"{label} {i}: ({x1},{y1}),({x2},{y2})"

        # Draw text above the box
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_weapon_boxes(frame, boxes, color=(0, 0, 255)):
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Weapon {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame