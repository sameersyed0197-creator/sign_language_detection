import cv2, os, numpy as np
from utils.hand_tracking import HandTracker
from utils.landmarks import normalize_landmarks_dual

SIGN_NAME = "i dont know"  # 🔴 CHANGE THIS EACH TIME
SAMPLES = 80
SAVE_DIR = f"data/dataset/{SIGN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
tracker = HandTracker(maxHands=2)

count, skip = 0, 0

while count < SAMPLES:
    ret, img = cap.read()
    if not ret:
        break

    img = tracker.findHands(img)
    hands = tracker.getLandmarks()

    if len(hands) >= 1:
        hand1 = hands[0]
        hand2 = hands[1] if len(hands) > 1 else None

        features = normalize_landmarks_dual(hand1, hand2)

        # 🔥 FIXED LINE:
        if features is not None and len(features) == 126:
            if skip >= 8:
                np.save(f"{SAVE_DIR}/{count}.npy", features)
                print(f"✅ Saved {count+1}/{SAMPLES}")
                count += 1
                skip = 0
            else:
                skip += 1

    cv2.putText(img, f"{SIGN_NAME} {count}/{SAMPLES}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collecting", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Dataset complete! Saved {count} samples to {SAVE_DIR}")











# import cv2, os, numpy as np
# from utils.hand_tracking import HandTracker
# from utils.landmarks import normalize_landmarks_dual
#
# SIGN_NAME = "nice to meet you"  # 🔴 CHANGE THIS EACH TIME
# SAMPLES = 80
# SAVE_DIR = f"data/dataset/{SIGN_NAME}"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# cap = cv2.VideoCapture(0)
# tracker = HandTracker(maxHands=2)
#
# count, skip = 0, 0
#
# while count < SAMPLES:
#     ret, img = cap.read()
#     if not ret:
#         break
#
#     img = tracker.findHands(img)
#     hands = tracker.getLandmarks()
#
#     if len(hands) >= 1:
#         hand1 = hands[0]
#         hand2 = hands[1] if len(hands) > 1 else None
#
#         features = normalize_landmarks_dual(hand1, hand2)
#
#         if features and len(features) == 126:
#             if skip >= 8:
#                 np.save(f"{SAVE_DIR}/{count}.npy", features)
#                 print(f"✅ Saved {count+1}/{SAMPLES}")
#                 count += 1
#                 skip = 0
#             else:
#                 skip += 1
#
#     cv2.putText(img, f"{SIGN_NAME} {count}/{SAMPLES}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     cv2.imshow("Collecting", img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#












# import cv2
# import os
# import numpy as np
# from utils.hand_tracking import HandTracker
# from utils.landmarks import normalize_landmarks
#
# # ================= CONFIG =================
# SIGN_NAME = "i love you"     # 🔴 CHANGE THIS EACH TIME
# SAMPLES = 40            # 30–50 recommended
# DATA_DIR = "data/dataset"
# # =========================================
#
# SAVE_PATH = os.path.join(DATA_DIR, SIGN_NAME)
# os.makedirs(SAVE_PATH, exist_ok=True)
#
# cap = cv2.VideoCapture(0)
# tracker = HandTracker()
#
# count = 0
# print(f"🟢 Collecting landmark data for sign: {SIGN_NAME}")
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     img = tracker.findHands(img)
#     lmList = tracker.findLandmarks(img)
#
#     if lmList:
#         features = normalize_landmarks(lmList)
#
#         if features is not None:
#             np.save(os.path.join(SAVE_PATH, f"{count}.npy"), features)
#             count += 1
#             print(f"Saved {count}/{SAMPLES}")
#
#     cv2.putText(
#         img,
#         f"{SIGN_NAME} : {count}/{SAMPLES}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2
#     )
#
#     cv2.imshow("Landmark Capture (.npy)", img)
#
#     if count >= SAMPLES:
#         break
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# print("✅ Landmark collection DONE")
