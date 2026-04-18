# Automated Defect Detection

A real-time camera tool for capturing and sorting product images into **pass** or **defect** categories, with a built-in machine learning classifier that predicts the result on every frozen frame.

## Requirements

- Python 3.8+
- Windows (uses DirectShow for camera access)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Program

```bash
python main.py
```

The program will scan for connected cameras on startup and open a 960×600 window showing the live feed.

---

## Workflow

### 1. Live View
The camera feed runs continuously. Use these controls:

| Key | Action |
|-----|--------|
| `SPACE` | Freeze the current frame |
| `C` | Cycle to the next available camera |
| `Q` / `ESC` | Quit |

### 2. Naming (after freezing)
The frame is frozen and a filename text box becomes active. If a trained model is loaded, the ML prediction (**PASS** or **DEFECT**) is shown on the frame.

| Key | Action |
|-----|--------|
| Type | Enter a filename |
| `Backspace` | Delete last character |
| `ENTER` | Confirm the filename and proceed to sorting |
| `ESC` | Discard and resume live feed |

The saved filename will be prefixed with a timestamp: `YYYYMMDD_HHMMSS_<your_name>.png`

### 3. Sorting
Choose whether the captured image is a passing or defective part.

| Key | Action |
|-----|--------|
| `P` | Save to `captures/pass/` |
| `D` | Save to `captures/defect/` |
| `R` | Go back to rename the file |
| `ESC` | Discard and resume live feed |

## Output

Captured images are saved under the `captures/` directory:

```
captures/
    pass/       # images sorted as passing
    defect/     # images sorted as defective
```

Each file is named: `YYYYMMDD_HHMMSS_<filename>.png`

---

## ML Model

### How it works

The classifier uses a two-stage feature extraction pipeline fed into a Support Vector Machine (SVM):

1. **HOG (Histogram of Oriented Gradients)** — captures the shape and texture of the part by computing gradient orientations across 8×8 pixel cells. This is effective at detecting surface irregularities, cracks, and geometric deformations.

2. **HSV color histograms** — captures the colour distribution across hue, saturation, and value channels. Useful for detecting discolouration, burns, or contamination.

These features are normalised with `StandardScaler` and classified by an RBF-kernel SVM (`C=10`, `gamma="scale"`). The SVM outputs a probability score used to display confidence alongside the PASS/DEFECT label.

### Training

Once you have collected sorted images via the main program, run:

```bash
python train_model.py
```

The script will:
- Load all images from `captures/pass/` and `captures/defect/`
- Extract HOG + colour histogram features from each image
- Evaluate accuracy using leave-one-out CV (small datasets) or 5-fold CV (10+ images)
- Train a final SVM on the full dataset
- Save the model to `defect_model.pkl`

Relaunch `main.py` after training — it loads the model at startup and will show predictions on every frozen frame.

### Tips for better accuracy

| Tip | Why |
|-----|-----|
| Collect 20–30+ images per class | SVM accuracy is unreliable below ~10 per class |
| Use consistent lighting | HOG and colour features are sensitive to lighting changes |
| Keep the camera angle fixed | Geometric consistency improves HOG features |
| Retrain after adding new images | The model is static until you run `train_model.py` again |
