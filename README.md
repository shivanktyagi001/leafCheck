# LeafCheck–CropDisease Detection Using Deep Learning and MLOps

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11%2B-blueviolet.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.48%2B-purple.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Artificial Intelligence and MLOps system that detects and diagnoses tomato leaf diseases from uploaded images with statistical confidence, top-3 candidates, and agricultural recommendations.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement & Objectives](#2-problem-statement--objectives)
- [3. System Architecture](#3-system-architecture)
- [4. Supported Tomato Disease Classes](#4-supported-tomato-disease-classes)
- [5. Technology Stack](#5-technology-stack)
- [6. Deep Learning & Transfer Learning Pipeline](#6-deep-learning--transfer-learning-pipeline)
- [7. MLOps Lifecycle](#7-mlops-lifecycle)
- [8. REST API Endpoints & Swagger UI](#8-rest-api-endpoints--swagger-ui)
- [9. Docker Containerization](#9-docker-containerization)
- [10. Automated Testing](#10-automated-testing)
- [11. Model & Data Monitoring](#11-model--data-monitoring)
- [12. Quick Start Guide](#12-quick-start-guide)

---

## 1. Project Overview
**LeafCheck** bridges cutting-edge Computer Vision and production MLOps engineering. Farmers and agronomists take a photo of a diseased or suspected tomato leaf, upload it to the system, and receive an instant diagnosis along with practical biological and chemical remedies.

### Key Highlights
- **Pretrained Deep Learning:** Powered by **MobileNetV2** (ImageNet pretraining) fine-tuned on the PlantVillage tomato corpus.
- **Production REST API:** Built with **FastAPI**, featuring async I/O, Pydantic data schemas, and interactive Swagger documentation at `/docs`.
- **Top-3 Ranking & Explainability:** Provides probability distributions across top candidates to prevent blind trust in single predictions.
- **Agricultural Decision Support:** Automatically enriches predictions with causes, visual symptoms, cultural prevention, and chemical/biological remedies.
- **Full MLOps Pipeline:** Experiment tracking via **MLflow**, data and pipeline version control via **DVC**, CI/CD via **GitHub Actions**, and drift observability via **Evidently AI**.
- **Dockerized:** Containerized for zero-friction reproducibility and cloud deployment.

---

## 2. Problem Statement & Objectives

### Problem Statement
Tomato (*Solanum lycopersicum*) is one of the world's most widely consumed agricultural commodities. However, yield losses exceeding 40–70% frequently occur due to fungal pathogens (Early Blight, Late Blight), bacterial infections (Bacterial Spot), viral vectors (Tomato Yellow Leaf Curl Virus), and pest infestations (Spider Mites). Traditional agricultural extension services are slow, costly, and scarce in rural regions, leading to delayed diagnoses or excessive, misdirected pesticide spraying.

### Objectives
1. Build an automated image classification model classifying 10 tomato leaf conditions.
2. Employ Transfer Learning using depthwise separable convolutions (MobileNetV2) for ultra-fast, low-latency inference on standard CPU instances.
3. Establish reproducible MLOps pipelines (versioning data, tracking experiments, logging metrics).
4. Containerize the application using Docker and provide interactive Swagger API testing.
5. Implement automated drift detection to observe distribution shifts in production inference.

---

## 3. System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 LEAFCHECK SYSTEM                                  |
+-----------------------------------------------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
      [ DATA & MLOps PIPELINE ]                           [ SERVING RUNTIME ]
                |                                                   |
    +-----------------------+                             +--------------------+
    | PlantVillage Dataset  |                             |  Client / Swagger  |
    +-----------------------+                             +--------------------+
                |                                                   | HTTP POST /predict
    +-----------------------+                             +--------------------+
    | DVC Pipeline Tracking |                             |  FastAPI Gateway   |
    +-----------------------+                             +--------------------+
                |                                                   |
    +-----------------------+                             +--------------------+
    | train.py (MobileNetV2)|                             | Input Validation   |
    | - Stage 1: Frozen Base|                             | (Size, MIME, PIL)  |
    | - Stage 2: Fine-Tuning|                             +--------------------+
    +-----------------------+                                       |
          /          \                                    +--------------------+
         v            v                                   | Preprocess Image   |
  +------------+  +----------------------+                | (224x224, [-1, 1]) |
  | MLflow     |  | models/              |                +--------------------+
  | Tracking & |  | leafcheck_model.keras|                          |
  | Artifacts  |  | classes.json         |--------(Loads Weights)---+
  +------------+  +----------------------+                          |
                                                          +--------------------+
                                                          | MobileNetV2 Engine |
                                                          +--------------------+
                                                                    |
                                                          +--------------------+
                                                          | Top-3 Softmax Rank |
                                                          +--------------------+
                                                                    |
                                                          +--------------------+
                                                          | Knowledge Base     |
                                                          | (Symptoms/Cure)    |
                                                          +--------------------+
                                                                    |
                                                          +--------------------+
                                                          | JSON Response      |
                                                          +--------------------+
```

---

## 4. Supported Tomato Disease Classes

| Class Label | Common Name | Pathogen Type | Typical Symptoms |
|:---|:---|:---|:---|
| `Tomato___Bacterial_spot` | Bacterial Spot | Bacterium (*Xanthomonas*) | Dark brown/black spots with yellow halo |
| `Tomato___Early_blight` | Early Blight | Fungus (*Alternaria solani*) | Concentric "target-board" rings on lower foliage |
| `Tomato___Late_blight` | Late Blight | Water mold (*Phytophthora infestans*) | Water-soaked dark lesions; white mold in humidity |
| `Tomato___Leaf_Mold` | Leaf Mold | Fungus (*Passalora fulva*) | Pale chlorotic spots with olive-green underside mold |
| `Tomato___Septoria_leaf_spot` | Septoria Leaf Spot | Fungus (*Septoria lycopersici*) | Small circular spots with gray centers and black specks |
| `Tomato___Spider_mites...` | Spider Mite Damage | Pest (*Tetranychus urticae*) | Fine stippling, leaf bronzing, fine silk webbing |
| `Tomato___Target_Spot` | Target Spot | Fungus (*Corynespora cassiicola*) | Circular lesions with concentric rings and defoliation |
| `Tomato___Tomato_Yellow_Leaf...`| Yellow Leaf Curl | Virus (*TYLCV* via Whitefly) | Severe upward curling, stunted growth, margin yellowing |
| `Tomato___Tomato_mosaic_virus` | Tomato Mosaic | Virus (*ToMV* via contact) | Mottling of green patterns, blistered/distorted leaves |
| `Tomato___healthy` | Healthy Leaf | None | Uniform green leaf blade, normal turgor, no lesions |

---

## 5. Technology Stack

- **Deep Learning Framework:** TensorFlow 2.16+ & Keras 3
- **Base Neural Architecture:** MobileNetV2 (ImageNet weights)
- **Scientific Computing:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **API Framework:** FastAPI, Uvicorn (ASGI), Pydantic v2
- **Image Processing:** Pillow (PIL)
- **Experiment Tracking:** MLflow
- **Data & Pipeline Versioning:** DVC (Data Version Control)
- **Model Observability:** Evidently AI
- **Containerization:** Docker (Multi-stage Python 3.12-slim)
- **CI/CD:** GitHub Actions

---

## 6. Deep Learning & Transfer Learning Pipeline

### Why MobileNetV2?
1. **Depthwise Separable Convolutions:** Decomposes a standard convolution into spatial filtering (depthwise) followed by channel mixing (pointwise). Reduces parameter count and floating-point operations by ~8× compared to traditional CNNs.
2. **Inverted Residuals & Linear Bottlenecks:** Preserves representation manifolds across thin bottleneck layers.
3. **Efficiency:** Compact model footprint (~14 MB) enabling sub-50ms CPU inferences in containerized environments.

### Two-Stage Training Process
- **Stage 1 (Feature Extraction):** MobileNetV2 convolutional base is frozen (`trainable = False`). Only the newly attached Classification Head (GlobalAveragePooling2D, BatchNormalization, Dropout 0.3, Dense 10 with Softmax) is trained with an Adam optimizer ($lr = 10^{-3}$).
- **Stage 2 (Fine-Tuning):** The top 20 layers of the base network are unfrozen and trained jointly at an ultra-low learning rate ($lr = 5 \times 10^{-5}$) to adapt high-level feature extractors specifically to tomato leaf pathologies without destroying pretrained weights.

---

## 7. MLOps Lifecycle

### MLflow Tracking
Run the MLflow dashboard:
```powershell
mlflow ui --port 5000
```
Tracks:
- Hyperparameters: Learning rate, batch size, epochs, dropout, optimizer.
- Epoch-by-epoch training and validation loss/accuracy curves.
- Artifacts: Saved `.keras` weights, `classes.json`, confusion matrices.

### DVC Pipeline Reproducibility
```powershell
dvc repro
```
Executes the defined `dvc.yaml` stages:
1. `prepare`: Stratified data split (70% Train, 15% Val, 15% Test) with deduplication.
2. `train`: Two-stage model training.
3. `evaluate`: Test dataset evaluation and metric generation.

---

## 8. REST API Endpoints & Swagger UI

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | API status and route discovery |
| `GET` | `/health` | Model health, architecture, and loaded classes |
| `GET` | `/classes` | List of 10 supported tomato disease classes |
| `POST`| `/predict` | Multipart leaf image upload returning top-3 predictions and remedies |

### Interactive Documentation
Open in browser:
```
http://localhost:8000/docs
```

### Sample Prediction Response (`POST /predict`)
```json
{
  "success": true,
  "filename": "tomato_early_blight_sample.jpg",
  "predicted_class": "Tomato___Early_blight",
  "display_name": "Tomato Early Blight",
  "confidence": 0.9412,
  "confidence_percentage": "94.12%",
  "top_3_predictions": [
    {
      "class_name": "Tomato___Early_blight",
      "display_name": "Tomato Early Blight",
      "confidence": 0.9412,
      "percentage": "94.12%"
    },
    {
      "class_name": "Tomato___Target_Spot",
      "display_name": "Tomato Target Spot",
      "confidence": 0.0381,
      "percentage": "3.81%"
    },
    {
      "class_name": "Tomato___Bacterial_spot",
      "display_name": "Tomato Bacterial Spot",
      "confidence": 0.0102,
      "percentage": "1.02%"
    }
  ],
  "disease_info": {
    "disease_name": "Tomato Early Blight",
    "causal_agent": "Alternaria solani (Fungus)",
    "symptoms": "Dark brown to black lesions showing distinctive concentric rings (target-board appearance)...",
    "prevention": [
      "Mulch heavily around the base of plants to prevent fungal spores from splashing.",
      "Provide adequate plant spacing to maximize airflow.",
      "Stake and prune plants to keep foliage off the ground."
    ],
    "treatment": [
      "Apply protective fungicides like Chlorothalonil or Copper sulfate at first sign of infection.",
      "Use bio-fungicides based on Trichoderma harzianum."
    ]
  },
  "inference_time_ms": 38.45
}
```

---

## 9. Docker Containerization

### Build Image
```powershell
docker build -t leafcheck-api:latest .
```

### Run Container
```powershell
docker run -d -p 8000:8000 --name leafcheck-app leafcheck-api:latest
```

### Container Operations
```powershell
# View logs
docker logs -f leafcheck-app

# Inspect running containers
docker ps

# Stop & remove container
docker stop leafcheck-app
docker rm leafcheck-app
```

---

## 10. Automated Testing
Run the complete test suite with `pytest`:
```powershell
pytest tests/ -v
```
Covers:
- Root, health, and classes endpoint integrity.
- Image byte validation and normalization range $[-1, 1]$.
- Valid image classification and top-3 parsing.
- Corrupted file rejection (HTTP 400).
- Unsupported MIME type rejection (HTTP 400).
- Payload size threshold protection (HTTP 413).

---

## 11. Model & Data Monitoring
Generate Evidently AI drift reports comparing baseline validation statistics to live inference logs:
```powershell
python monitoring/monitor.py
```
Outputs an interactive report at `monitoring/drift_report.html` evaluating:
- **Prediction Drift:** Shifts in disease frequency.
- **Confidence Drift:** Degradation in mean model certainty.
- **Latency Drift:** Inference response time anomalies.

### Prepare a new labeled image batch (Task A)
Place incoming images under `dataset/new_batch/`, with one folder per target class
using the class names in `models/classes.json`:
```text
dataset/new_batch/
├── Tomato___Early_blight/
│   └── new_leaf_001.jpg
└── Tomato___healthy/
    └── new_leaf_002.jpg
```
Validate the labels and image files, check duplicates by image content, and create
a size and class-distribution report:
```powershell
python src/prepare_new_data.py
```
The command prints the report and saves it to
`monitoring/new_batch_data_report.json`. Use `--batch-dir`, `--classes-file`, and
`--report-path` to override the defaults. Invalid or unlabeled files are listed
in the report and cause a non-zero exit status; duplicate copies are reported
and excluded from the unique-image class counts.

### Evaluate the existing model on a labeled batch (Task B)
Run evaluation against the new batch after placing its labeled images in
`dataset/new_batch/`:
```powershell
python src/evaluate.py --data-dir dataset/new_batch --output-dir monitoring/new_batch
```
The saved model in `models/leafcheck_model.keras` is evaluated without retraining.
The JSON report at `monitoring/new_batch/classification_report.json` records the
evaluated dataset and sample count, overall accuracy, weighted precision, recall
and F1-score, plus per-class metrics. The confusion matrix is saved beside it.
Weighted precision, recall and F1 account for class imbalance in the labeled
batch, while accuracy gives an overall correctness measure. Pass `--model-path`,
`--classes-file`, `--data-dir`, `--output-dir`, or `--report-path` to use
different inputs or artifact locations. Without arguments, the existing
`dataset/test` evaluation used by DVC remains the default.

### Retrain a separate candidate with the labeled batch (Task C)
After validating and evaluating the batch, train a fresh candidate using both
`dataset/train/` and `dataset/new_batch/`. The existing validation split is still
used to monitor training:
```powershell
python src/train.py --new-data-dir dataset/new_batch
```
The candidate model and matching class mapping are saved separately at
`models/candidate/leafcheck_candidate.keras` and `models/candidate/classes.json`.
The existing champion at `models/leafcheck_model.keras` is not replaced. Use
`--candidate-model-path` and `--candidate-classes-path` to choose different
candidate output paths; paths resolving to the champion artifacts are rejected.
This uses the same MobileNetV2 preprocessing, augmentation, and two-stage
training pipeline as the existing training command, starting a fresh model
rather than continuing champion weights. The run is logged to the configured
MLflow experiment as `candidate_retraining`, including the added batch location
and combined training data locations.

### Complete workflow and quality gate
Run the complete labeled-batch workflow after the batch has been prepared:
```powershell
python src/run_workflow.py --batch-dir dataset/new_batch --quick
```
The workflow validates the batch, evaluates the champion, trains and evaluates
an isolated candidate, then compares champion and candidate accuracy and
weighted F1. It writes:

- `monitoring/workflow/workflow_result.json` with the full execution result.
- `monitoring/workflow/champion/classification_report.json` with the baseline evaluation.
- `monitoring/workflow/candidate/classification_report.json` with the candidate evaluation.
- `models/candidate/leafcheck_candidate.keras` and `models/candidate/classes.json` as candidate artifacts.

The quality gate approves only when both candidate metrics meet the champion
metrics (or the configured `--minimum-improvement` threshold). A rejected gate
returns a non-zero exit status; the champion artifacts are never overwritten.

---

## 12. Quick Start Guide

```powershell
# 1. Clone repository
git clone https://github.com/your-username/LeafCheck.git
cd LeafCheck

# 2. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Prepare dataset & run training
python src/data_setup.py
python src/train.py

# 5. Evaluate model
python src/evaluate.py

# 6. Start FastAPI server
uvicorn src.main:app --reload --port 8000
```
---
