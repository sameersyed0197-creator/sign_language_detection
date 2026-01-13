# import cv2
# import mediapipe as mp
#
# class HandTracker:
#     def __init__(self, maxHands=2, detectionCon=0.7, trackCon=0.7):
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(
#             max_num_hands=maxHands,
#             min_detection_confidence=detectionCon,
#             min_tracking_confidence=trackCon
#         )
#         self.mpDraw = mp.solutions.drawing_utils
#         self.results = None
#
#     def findHands(self, img):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 self.mpDraw.draw_landmarks(
#                     img, handLms, self.mpHands.HAND_CONNECTIONS
#                 )
#         return img
#
#     def getLandmarks(self):
#         hands = []
#
#         if self.results and self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 hand = []
#                 for lm in handLms.landmark:
#                     hand.append([lm.x, lm.y, lm.z])
#                 hands.append(hand)
#
#         return hands  # list of hands
#









import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, maxHands=2, detectionCon=0.6, trackCon=0.6):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,        # 🔥 enables tracking
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.results = None

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return img

    def getLandmarks(self):
        """
        Returns hands in consistent left-to-right order
        """
        hands = []
        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                hand = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
                hands.append(hand)

            # 🔥 consistent ordering (important for ML)
            hands.sort(key=lambda h: h[0][0])  # wrist x

        return hands

