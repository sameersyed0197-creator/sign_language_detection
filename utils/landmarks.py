# import numpy as np
#
# def normalize_landmarks_dual(hand1, hand2=None):
#     """
#     hand1, hand2: list of [x,y,z] landmarks (length 21)
#     Returns: 126 features (2 hands × 21 × 3)
#     """
#
#     def process(hand):
#         hand = np.array(hand)
#
#         # Wrist as origin
#         origin = hand[0]
#         hand = hand - origin
#
#         max_val = np.max(np.abs(hand))
#         if max_val == 0:
#             return None
#
#         hand = hand / max_val
#         return hand.flatten()
#
#     f1 = process(hand1)
#     if f1 is None:
#         return None
#
#     if hand2:
#         f2 = process(hand2)
#         if f2 is None:
#             return None
#     else:
#         # Pad second hand with zeros if not visible
#         f2 = np.zeros(63)
#
#     return np.concatenate([f1, f2]).tolist()
#
#
#
#
#
#
#
#
#
#
#
#
#
#












import numpy as np

def normalize_landmarks_dual(hand1, hand2=None):
    """
    Normalize dual-hand landmarks
    Returns 126 features (2 × 21 × 3)
    """

    def process(hand):
        hand = np.array(hand, dtype=np.float32)

        # Wrist as origin
        origin = hand[0]
        hand = hand - origin

        max_val = np.max(np.abs(hand))
        if max_val < 1e-6:
            return None

        hand = hand / max_val
        return hand.flatten()

    f1 = process(hand1)
    if f1 is None:
        return None

    if hand2 is not None:
        f2 = process(hand2)
        if f2 is None:
            return None
    else:
        f2 = np.zeros(63, dtype=np.float32)

    return np.concatenate([f1, f2])

