import numpy as np

def normalize_landmarks(landmarks):
    if len(landmarks) == 0:
        return None

    base_x, base_y = landmarks[0][1], landmarks[0][2]
    normalized = []

    for lm in landmarks:
        normalized.append([
            lm[1] - base_x,
            lm[2] - base_y
        ])

    return np.array(normalized).flatten()
