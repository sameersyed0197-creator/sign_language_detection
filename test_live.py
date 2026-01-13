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
FRAME_SKIP = 2

SIGN_HOLD_TIME = 0.6
# =========================================

# ================= LANGUAGE KEYS =================
LANG_KEYS = {
    ord('1'): ("en", "English"),
    ord('2'): ("hi", "Hindi"),
    ord('3'): ("te", "Telugu"),
    ord('4'): ("ta", "Tamil"),
}

# Default language set to English
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

print("🎥 Sign Language Detection - Landmark-Only Mode (FAST)")
print("🎛  Press 1 for English | 2 for Hindi | 3 for Telugu | 4 for Tamil | q to Quit")
print(f"🌍 Current Language: {current_lang_name}")

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1
    detected_sign = ""

    # ========= LANDMARK-ONLY PROCESS =========
    if frame_count % FRAME_SKIP == 0:
        tracker.findHands(img)
        hands = tracker.getLandmarks()

        # No hands detected → reset buffer
        if not hands:
            prediction_buffer.clear()
            last_pred_idx = None

        else:
            # Ensure minimum landmark quality
            if len(hands[0]) >= 21:
                features = normalize_landmarks_dual(
                    hands[0],
                    hands[1] if len(hands) > 1 else None
                )

                if features is not None:
                    probs = model.predict_proba([features])[0]
                    idx = np.argmax(probs)

                    if probs[idx] >= CONFIDENCE_THRESHOLD:

                        # New label detected → clear old buffer
                        if last_pred_idx is not None and idx != last_pred_idx:
                            prediction_buffer.clear()

                        prediction_buffer.append(idx)
                        if len(prediction_buffer) > BUFFER_SIZE:
                            prediction_buffer.pop(0)

                        # Require consistent predictions
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
                if detected_sign in LANGUAGE_MAP:
                    display_text = LANGUAGE_MAP[detected_sign][current_lang]
                    audio.speak(display_text)

                    last_detected_sign = detected_sign
                    last_update_time = now

    # ================= DISPLAY =================
    # Language selector info
    cv2.putText(
        img,
        f"Language: {current_lang_name} | Press 1-4 to change",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Current detected sign text
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

    # Language key guide
    cv2.putText(
        img,
        "1-English | 2-Hindi | 3-Telugu | 4-Tamil",
        (20, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    cv2.imshow("Sign to Speech - Landmark Detection", img)

    # ================= KEYBOARD CONTROLS =================
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        print("👋 Exiting...")
        break

    # Language switching
    if key in LANG_KEYS:
        current_lang, current_lang_name = LANG_KEYS[key]

        # Update display text to new language
        if last_detected_sign in LANGUAGE_MAP:
            display_text = LANGUAGE_MAP[last_detected_sign][current_lang]
            audio.speak(display_text)  # Speak in new language

        print(f"🌍 Switched to {current_lang_name}")

cap.release()
cv2.destroyAllWindows()
audio.cleanup()  # Clean up audio resources
print("✅ Program ended successfully")

























