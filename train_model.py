import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "data/dataset"
MODEL_PATH = "models/sign_model.pkl"

X = []
y = []
labels = []

for label_id, sign in enumerate(os.listdir(DATASET_PATH)):
    sign_path = os.path.join(DATASET_PATH, sign)
    if not os.path.isdir(sign_path):
        continue

    labels.append(sign)

    for file in os.listdir(sign_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(sign_path, file))
            X.append(data)
            y.append(label_id)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Total signs:", len(labels))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("✅ Model Accuracy:", acc)

os.makedirs("models", exist_ok=True)
joblib.dump((model, labels), MODEL_PATH)
print("✅ Model saved:", MODEL_PATH)
