# DeepFakeGuard---AI-Powered-Video-DeepFake-Detection
A deep learning pipeline that detects deepfake videos by combining ResNet50 spatial feature extraction with a Bidirectional LSTM to classify REAL vs FAKE based on facial frame sequences.


# DeepFake Video Detector 🎭🔍

A deep learning pipeline for detecting deepfake videos using **face-focused frame extraction**, **ResNet50 feature embeddings**, and a **Bidirectional LSTM (BiLSTM)** classifier to distinguish REAL videos from FAKE (deepfake) videos.

---

## 📌 Overview

Deepfakes are synthetically generated or manipulated videos that can convincingly swap or alter a person's face. This project builds an end-to-end video classification system that:

1. Extracts uniformly sampled frames from a video
2. Detects and crops the face region in each frame (Haar Cascade)
3. Passes each face frame through a pretrained **ResNet50** (ImageNet weights) to obtain spatial feature embeddings
4. Feeds the resulting frame-sequence embeddings into a **Bidirectional LSTM** network to learn temporal patterns across frames
5. Outputs a binary prediction: **REAL** or **FAKE**, along with a confidence score

This approach treats deepfake detection as a **spatio-temporal sequence classification problem** — combining CNN-based per-frame visual features with LSTM-based temporal modeling, since deepfake artifacts often manifest as subtle frame-to-frame inconsistencies.

---

## 🧠 Model Architecture

```
Video File
    │
    ▼
Frame Extraction (30 frames, uniformly sampled)
    │
    ▼
Face Detection & Cropping (Haar Cascade, 224x224)
    │
    ▼
ResNet50 (frozen, ImageNet weights) → GlobalAveragePooling
    │
    ▼
Feature Sequence (30 x 2048)
    │
    ▼
Bidirectional LSTM (128 units) → Dropout
    │
    ▼
Bidirectional LSTM (64 units) → Dropout
    │
    ▼
Dense (128, ReLU) → Dropout
    │
    ▼
Dense (1, Sigmoid) → REAL / FAKE
```

---

## 📂 Project Structure

```
DeepFake-Video-Detector/
│
├── DeepFake_Detection_Better.ipynb   # Main notebook (full pipeline)
├── video_features_resnet.npy         # Extracted ResNet50 features (generated)
├── video_labels.npy                  # Corresponding labels (generated)
├── Deep_Fake_Video_Detection.h5      # Trained BiLSTM model (generated)
├── training_history.png              # Accuracy/Loss training curves
└── README.md
```

> **Note:** The `.npy` feature files and `.h5` model weights are large binary artifacts generated during training/feature extraction. They are not tracked in this repo by default — see [Setup](#-setup--usage) below to regenerate them, or add them via Git LFS if you want to version them.

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/DeepFake-Video-Detector.git
cd DeepFake-Video-Detector
```

### 2. Install dependencies
```bash
pip install tensorflow opencv-python numpy pandas matplotlib tqdm scikit-learn seaborn mediapipe ultralytics
```

### 3. Prepare your dataset
Organize your video dataset into two folders:
```
CombinedDataset/
├── Real/
│   ├── video1.mp4
│   └── ...
└── Fake/
    ├── video1.mp4
    └── ...
```
Update the `DATASET_DIR` path in the notebook's configuration cell to point to your dataset location.

### 4. Run the notebook
Open `DeepFake_Detection_Better.ipynb` and run the cells in order:
1. **Feature Extraction** — extracts faces from each video and computes ResNet50 embeddings (saved as `.npy` files)
2. **LSTM Training** — trains the BiLSTM classifier on the extracted features
3. **Evaluation** — generates accuracy/loss plots, confusion matrix, and a classification report
4. **Inference** — run `predict_video("path/to/video.mp4")` to classify a new video

### Example inference
```python
predict_video("REAL_sample.mp4")
# Output: Label: REAL | Confidence: 0.92
```

---

## 📊 Evaluation

The notebook produces:
- **Training curves** (accuracy & loss over epochs) → `training_history.png`
- **Confusion matrix** comparing predicted vs. true labels
- **Classification report** (precision, recall, F1-score for REAL/FAKE classes)
- **Overall test accuracy**

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Deep Learning Framework | TensorFlow / Keras |
| Feature Extractor (CNN) | ResNet50 (pretrained on ImageNet) |
| Temporal Model | Bidirectional LSTM |
| Face Detection | OpenCV Haar Cascade |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |

---

## 🚧 Known Limitations / Future Work

- Face detection currently uses Haar Cascade, which is fast but less robust than DNN-based detectors (e.g., MTCNN, MediaPipe, or YOLO face models — already imported in the notebook and worth integrating fully).
- Fixed frame sampling (30 frames per video) may miss short-duration manipulation artifacts in longer videos.
- Model performance is dependent on dataset quality/diversity; testing on out-of-distribution deepfake generation methods (e.g., diffusion-based fakes) is recommended.
- Dataset paths are currently hardcoded for local use — consider parameterizing via config file or CLI args for portability.

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [ResNet50 — Keras Applications](https://keras.io/api/applications/resnet/)
- [OpenCV Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- Public deepfake detection datasets (e.g., FaceForensics++, DFDC) used for training/evaluation
