import cv2
import joblib
import numpy as np
import time

from utils.hand_tracking import HandTracker
from utils.landmarks import normalize_landmarks_dual
from audio_manager import AudioManager
from language_map import LANGUAGE_MAP

# ================= CONFIG =================
MODEL_PATH = "models/sign_dual_model.pkl"

CONFIDENCE_THRESHOLD = 0.55
BUFFER_SIZE = 3
MATCH_COUNT = 2
FRAME_SKIP = 2            # inference rate

SIGN_HOLD_TIME = 0.6      # seconds
# =========================================

# ================= LANGUAGE KEYS =================
LANG_KEYS = {
    ord('1'): ("en", "English"),
    ord('2'): ("hi", "Hindi"),
    ord('3'): ("te", "Telugu"),
    ord('4'): ("ta", "Tamil"),
}

current_lang = "en"
current_lang_name = "English"

# ================= LOAD MODEL =================
model, labels = joblib.load(MODEL_PATH)
tracker = HandTracker(maxHands=2)
audio = AudioManager(rate=160, gap=1.2)

# ================= STATE =================
prediction_buffer = []
frame_count = 0

last_detected_sign = ""
display_text = ""
last_update_time = 0
last_pred_idx = None

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)

print("🎥 Sign Language Detection (Tracking + Landmark Only)")
print("🎛 1-English | 2-Hindi | 3-Telugu | 4-Tamil | q-Quit")

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1
    detected_sign = ""

    # 🔥 ALWAYS TRACK (every frame)
    tracker.findHands(img)

    # 🔥 PREDICT only every N frames
    if frame_count % FRAME_SKIP == 0:
        hands = tracker.getLandmarks()

        # No hands → reset
        if not hands:
            prediction_buffer.clear()
            last_pred_idx = None

        else:
            features = normalize_landmarks_dual(
                hands[0],
                hands[1] if len(hands) > 1 else None
            )

            if features is not None:
                probs = model.predict_proba([features])[0]
                idx = np.argmax(probs)

                if probs[idx] >= CONFIDENCE_THRESHOLD:

                    # New sign → reset buffer
                    if last_pred_idx is not None and idx != last_pred_idx:
                        prediction_buffer.clear()

                    prediction_buffer.append(idx)
                    if len(prediction_buffer) > BUFFER_SIZE:
                        prediction_buffer.pop(0)

                    if prediction_buffer.count(idx) >= MATCH_COUNT:
                        detected_sign = labels[idx]

                    last_pred_idx = idx

    # ========= UPDATE DISPLAY + AUDIO =========
    now = time.time()
    if detected_sign:
        if (
                detected_sign != last_detected_sign
                or now - last_update_time > SIGN_HOLD_TIME
        ):
            display_text = LANGUAGE_MAP[detected_sign][current_lang]
            audio.speak(display_text)

            last_detected_sign = detected_sign
            last_update_time = now

    # ================= DISPLAY =================
    cv2.putText(
        img,
        f"Language: {current_lang_name} (1-4)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    if display_text:
        cv2.putText(
            img,
            display_text,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 255),
            3,
        )

    cv2.imshow("Sign → Speech (Final Stable)", img)

    # ================= KEYS =================
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key in LANG_KEYS:
        current_lang, current_lang_name = LANG_KEYS[key]
        if last_detected_sign in LANGUAGE_MAP:
            display_text = LANGUAGE_MAP[last_detected_sign][current_lang]
            audio.speak(display_text)

        print(f"🌍 Switched to {current_lang_name}")

cap.release()
cv2.destroyAllWindows()
audio.cleanup()
print("✅ Program ended successfully")
