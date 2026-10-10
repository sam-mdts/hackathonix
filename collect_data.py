import csv, os, sys
import cv2
import numpy as np
import mediapipe as mp

PERSON_ID = sys.argv[1] if len(sys.argv) > 1 else input("Person id (e.g. p01): ")
FRAMES_PER_PRESS = 60
OUT = "data.csv"

def normalize(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark])
    pts -= pts[0]                              # center on wrist
    scale = np.linalg.norm(pts[9]) or 1.0      # wrist -> middle knuckle
    return (pts / scale).flatten()             # 63 numbers

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

new_file = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow(["person", "label"] + [f"f{i}" for i in range(63)])

recording, label, count = False, None, 0
totals = {}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None

    if hand:
        draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        if recording:
            writer.writerow([PERSON_ID, label] + list(normalize(hand)))
            count += 1
            if count >= FRAMES_PER_PRESS:
                recording = False
                totals[label] = totals.get(label, 0) + count
                f.flush()

    status = f"REC {label} {count}/{FRAMES_PER_PRESS}" if recording else "Press a letter key to record (ESC quits)"
    cv2.putText(frame, f"{PERSON_ID} | {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, str(totals), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("collect", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if not recording and ord("a") <= key <= ord("z") and chr(key) not in "jz":
        recording, label, count = True, chr(key).upper(), 0

f.close()
cap.release()
cv2.destroyAllWindows()