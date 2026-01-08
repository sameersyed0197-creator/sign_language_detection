import cv2
import os
import numpy as np
from utils.hand_tracking import HandTracker
from utils.landmarks import normalize_landmarks

# ================= CONFIG =================
SIGN_NAME = "Hello"     # 🔴 CHANGE THIS EACH TIME
SAMPLES = 40            # 30–50 recommended
DATA_DIR = "data/dataset"
# =========================================

SAVE_PATH = os.path.join(DATA_DIR, SIGN_NAME)
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
tracker = HandTracker()

count = 0
print(f"🟢 Collecting landmark data for sign: {SIGN_NAME}")

while True:
    success, img = cap.read()
    if not success:
        break

    img = tracker.findHands(img)
    lmList = tracker.findLandmarks(img)

    if lmList:
        features = normalize_landmarks(lmList)

        if features is not None:
            np.save(os.path.join(SAVE_PATH, f"{count}.npy"), features)
            count += 1
            print(f"Saved {count}/{SAMPLES}")

    cv2.putText(
        img,
        f"{SIGN_NAME} : {count}/{SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Landmark Capture (.npy)", img)

    if count >= SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Landmark collection DONE")
