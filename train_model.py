import cv2
import numpy as np
import os
import sys
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "defect_model.pkl")
PASS_DIR = os.path.join(BASE_DIR, "captures", "pass")
DEFECT_DIR = os.path.join(BASE_DIR, "captures", "defect")
IMG_SIZE = 128


def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_desc = cv2.HOGDescriptor(
        _winSize=(IMG_SIZE, IMG_SIZE),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_feats = hog_desc.compute(gray).flatten()

    return np.concatenate([h_hist, s_hist, v_hist, hog_feats])


def load_dataset(directory, label):
    X, y = [], []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(os.path.join(directory, fname))
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(label)
    return X, y


def main():
    print("Loading images and extracting features...")
    pass_X, pass_y = load_dataset(PASS_DIR, 0)    # 0 = pass
    defect_X, defect_y = load_dataset(DEFECT_DIR, 1)  # 1 = defect

    n_pass, n_defect = len(pass_y), len(defect_y)
    if n_pass == 0 or n_defect == 0:
        print(f"Need images in both folders. Found: {n_pass} pass, {n_defect} defect.")
        print("Capture and sort images with main.py first.")
        sys.exit(1)

    print(f"  Pass:   {n_pass} images")
    print(f"  Defect: {n_defect} images")

    X = np.array(pass_X + defect_X)
    y = np.array(pass_y + defect_y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])

    n_total = len(y)
    if n_total >= 10:
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"\n5-fold CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    elif n_total >= 4:
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
        print(f"\nLeave-one-out accuracy: {scores.mean():.1%}  ({n_total} samples — collect more for reliable metrics)")
    else:
        print(f"\n{n_total} samples — skipping cross-validation, collect more images for accuracy estimates.")

    print("Training final model on all data...")
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
