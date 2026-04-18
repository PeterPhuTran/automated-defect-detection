import cv2
import numpy as np
from datetime import datetime
import os
import sys


CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "captures")
PASS_DIR = os.path.join(CAPTURES_DIR, "pass")
DEFECT_DIR = os.path.join(CAPTURES_DIR, "defect")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model.pkl")
IMG_SIZE = 128
FONT = cv2.FONT_HERSHEY_SIMPLEX

_model = None


def _load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        print("No model found — run train_model.py to train one.")
        return
    try:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print("Defect model loaded.")
    except Exception as e:
        print(f"Could not load model: {e}")


def _extract_features(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
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


def predict(frame):
    if _model is None:
        return None
    try:
        feats = _extract_features(frame).reshape(1, -1)
        label_idx = int(_model.predict(feats)[0])
        conf = float(_model.predict_proba(feats)[0][label_idx])
        return ("PASS" if label_idx == 0 else "DEFECT"), conf
    except Exception:
        return None


def _draw_prediction(frame, prediction):
    if prediction is None:
        label_text = "No model — run train_model.py"
        cv2.putText(frame, label_text, (10, 65), FONT, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
        return
    label, conf = prediction
    color = (0, 210, 0) if label == "PASS" else (0, 0, 220)
    cv2.putText(frame, f"{label}  {conf:.0%}", (10, 68), FONT, 1.0, color, 2, cv2.LINE_AA)


def find_cameras(max_index=10):
    available = []
    print("Scanning for cameras...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"  Found camera at index {i}")
            cap.release()
    return available


def draw_live_overlay(frame, cam_index, cam_list):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame,
                f"Camera {cam_index}  ({len(cam_list)} found)  |  [SPACE] Freeze  |  [C] Cycle camera  |  [Q] Quit",
                (10, 28), FONT, 0.55, (0, 230, 0), 1, cv2.LINE_AA)
    return frame


def draw_naming_overlay(frame, filename_input, frozen_time, prediction):
    h, w = frame.shape[:2]
    bar = frame.copy()
    panel_h = 90
    cv2.rectangle(bar, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.7, frame, 0.3, 0, frame)

    ts_str = frozen_time.strftime("%Y%m%d_%H%M%S")
    cv2.putText(frame, f"Prefix: {ts_str}", (10, h - panel_h + 22),
                FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Filename: {filename_input}|",
                (10, h - panel_h + 52), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "[ENTER] Confirm name    [ESC] Resume live feed",
                (10, h - 12), FONT, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "FROZEN", (w - 110, 30), FONT, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    _draw_prediction(frame, prediction)
    return frame


def draw_sorting_overlay(frame, filename_input, frozen_time, prediction):
    h, w = frame.shape[:2]
    bar = frame.copy()
    panel_h = 90
    cv2.rectangle(bar, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.7, frame, 0.3, 0, frame)

    ts_str = frozen_time.strftime("%Y%m%d_%H%M%S")
    safe_name = filename_input.strip().replace(" ", "_") or "capture"
    cv2.putText(frame, f"Saving as: {ts_str}_{safe_name}.png", (10, h - panel_h + 22),
                FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "Sort image:",
                (10, h - panel_h + 52), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "[P] Pass    [D] Defect    [R] Rename    [ESC] Resume live feed",
                (10, h - 12), FONT, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "FROZEN", (w - 110, 30), FONT, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    _draw_prediction(frame, prediction)
    return frame


def main():
    os.makedirs(PASS_DIR, exist_ok=True)
    os.makedirs(DEFECT_DIR, exist_ok=True)
    _load_model()

    cameras = find_cameras()
    if not cameras:
        print("No cameras found. Exiting.")
        sys.exit(1)

    print(f"Found {len(cameras)} camera(s): {cameras}")

    cam_idx = 0
    cap = cv2.VideoCapture(cameras[cam_idx], cv2.CAP_DSHOW)

    state = "live"  # "live" | "naming" | "sorting"
    frozen_frame = None
    frozen_time = None
    filename_input = ""
    prediction = None

    window = "Automated Defect Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 600)

    while True:
        if state == "live":
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed — trying to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(cameras[cam_idx], cv2.CAP_DSHOW)
                continue
            display = draw_live_overlay(frame.copy(), cameras[cam_idx], cameras)
        elif state == "naming":
            display = draw_naming_overlay(frozen_frame.copy(), filename_input, frozen_time, prediction)
        else:
            display = draw_sorting_overlay(frozen_frame.copy(), filename_input, frozen_time, prediction)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if state == "live":
            if key in (ord('q'), 27):       # Q or ESC → quit
                break
            elif key == ord(' '):           # Space → freeze
                state = "naming"
                frozen_frame = frame.copy()
                frozen_time = datetime.now()
                filename_input = ""
                prediction = predict(frozen_frame)
            elif key == ord('c'):           # C → cycle camera
                cam_idx = (cam_idx + 1) % len(cameras)
                cap.release()
                cap = cv2.VideoCapture(cameras[cam_idx], cv2.CAP_DSHOW)
                print(f"Switched to camera index {cameras[cam_idx]}")
        elif state == "naming":
            if key == 27:                   # ESC → resume live
                state = "live"
                filename_input = ""
                prediction = None
            elif key == 13:                 # Enter → confirm name, go to sorting
                state = "sorting"
            elif key == 8:                  # Backspace
                filename_input = filename_input[:-1]
            elif 32 <= key <= 126:          # Printable ASCII
                filename_input += chr(key)
        else:  # sorting
            if key == 27:                   # ESC → resume live
                state = "live"
                filename_input = ""
                prediction = None
            elif key == ord('r'):           # R → back to naming
                state = "naming"
            elif key in (ord('p'), ord('d')):  # P → pass, D → defect
                safe_name = filename_input.strip().replace(" ", "_") or "capture"
                timestamp = frozen_time.strftime("%Y%m%d_%H%M%S")
                folder = PASS_DIR if key == ord('p') else DEFECT_DIR
                label = "pass" if key == ord('p') else "defect"
                save_path = os.path.join(folder, f"{timestamp}_{safe_name}.png")
                cv2.imwrite(save_path, frozen_frame)
                print(f"Saved [{label}]: {save_path}")
                state = "live"
                filename_input = ""
                prediction = None

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
