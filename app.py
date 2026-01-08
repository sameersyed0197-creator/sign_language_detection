import cv2
import pyttsx3
from utils.hand_tracking import HandTracker
from utils.landmarks import normalize_landmarks

# Initialize
cap = cv2.VideoCapture(0)
tracker = HandTracker()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

last_spoken = ""

while True:
    success, img = cap.read()
    if not success:
        break

    img = tracker.findHands(img)
    lmList = tracker.findLandmarks(img)

    if lmList:
        features = normalize_landmarks(lmList)

        # TEMP RULE (placeholder)
        sign = "HELLO"

        if sign != last_spoken:
            engine.say(sign)
            engine.runAndWait()
            last_spoken = sign

        cv2.putText(
            img,
            f"Detected: {sign}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Sign Language Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
