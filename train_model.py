import os, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y, labels = [], [], []

DATASET = "data/dataset"

for idx, sign in enumerate(sorted(os.listdir(DATASET))):
    labels.append(sign)
    for f in os.listdir(f"{DATASET}/{sign}"):
        if f.endswith(".npy"):
            data = np.load(f"{DATASET}/{sign}/{f}")
            if len(data) == 126:
                X.append(data)
                y.append(idx)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Accuracy:", model.score(X_test, y_test) * 100)

joblib.dump((model, labels), "models/sign_dual_model.pkl")
print("✅ Model saved to models/sign_dual_model.pkl")