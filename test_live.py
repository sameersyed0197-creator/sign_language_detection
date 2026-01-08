import cv2
import joblib
from utils.hand_tracking import HandTracker
from utils.landmarks import normalize_landmarks

# Load trained model
model, labels = joblib.load("models/sign_model.pkl")

tracker = HandTracker()
cap = cv2.VideoCapture(0)

last_sign = ""

print("🟢 Live testing started... Press Q to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = tracker.findHands(img)
    lmList = tracker.findLandmarks(img)

    if lmList:
        features = normalize_landmarks(lmList)

        if features is not None:
            prediction = model.predict([features])[0]
            sign = labels[prediction]

            if sign != last_sign:
                print("Detected:", sign)
                last_sign = sign

            cv2.putText(
                img,
                f"Detected: {sign}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Live Sign Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
