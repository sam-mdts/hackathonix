import time
from collections import deque, Counter
import cv2
import joblib
import numpy as np
import mediapipe as mp

HOLD_SECONDS = 1   # how long to hold a letter before it counts
CONF_MIN = 0.5       # ignore guesses the model isn't sure about

model = joblib.load("model.pkl")
model.n_jobs = 1     # faster for one guess at a time

def normalize(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark])
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) or 1.0
    return (pts / scale).flatten()

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

word = ""
candidate = None
candidate_start = 0.0
locked = False
recent = deque(maxlen=5)
prev_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None

    guess = None
    if hand:
        draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        probs = model.predict_proba([normalize(hand)])[0]
        best = probs.argmax()
        if probs[best] >= CONF_MIN:
            guess = model.classes_[best]
    recent.append(guess)

    # Smoothing: a letter must win at least 3 of the last 5 frames
    counts = Counter(g for g in recent if g)
    letter = None
    if counts:
        top, n = counts.most_common(1)[0]
        if n >= 3:
            letter = top

    now = time.time()
    if letter != candidate:
        candidate = letter
        candidate_start = now
        locked = False
    elif candidate and not locked and now - candidate_start >= HOLD_SECONDS:
        word += candidate
        locked = True

    # FPS
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    # Drawing
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Letter: {letter or '-'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    if candidate and not locked:
        progress = min((now - candidate_start) / HOLD_SECONDS, 1.0)
        cv2.rectangle(frame, (10, 55), (10 + int(200 * progress), 70), (0, 255, 0), -1)
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, word, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE=space  BACKSPACE=delete  C=clear  ESC=quit",
                (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("ASL alphabet recognizer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        word += " "
    elif key == 8:
        word = word[:-1]
    elif key == ord("c"):
        word = ""

cap.release()
cv2.destroyAllWindows()