# SignBridge

**SignBridge** is an AI-powered educational assistant designed to support Deaf students through sign language recognition, course-grounded educational assistance, simplified explanations, and avatar-based sign presentation.

The project focuses primarily on **Jordanian Sign Language (JoSL)** within a technical educational context and combines locally collected Jordanian IT signs with the **KArSL** and **Isharah** datasets for multi-source sign-recognition experiments.

---

## Project Overview

SignBridge integrates several components into one modular educational pipeline:

```text
Sign Input
   ↓
Sign Recognition
   ↓
Recognized Sign / Gloss
   ↓
Course Knowledge Retrieval
   ↓
Educational Response Generation
   ↓
Simplified Educational Explanation
   ↓
Semantic Sign Planning
   ↓
Sign Tokens
   ↓
Motion Lookup
   ↓
3D Avatar Sign Presentation
```

The current research prototype investigates both **isolated-sign recognition** and **continuous sign recognition**, together with retrieval-augmented educational assistance and an experimental signed-output pipeline.

---

## Key Components

SignBridge includes:

- Jordanian IT-domain sign recognition
- isolated sign recognition
- continuous sign recognition
- signer-independent evaluation
- multi-source learning
- multi-task learning
- transfer learning
- knowledge distillation
- MediaPipe-based landmark extraction
- temporal sign modeling
- retrieval-augmented generation (`RAG`)
- educational content simplification
- semantic sign planning
- local sign-motion lookup
- Three.js-based avatar playback
- FastAPI backend
- Vite + Three.js frontend

---

## Repository Structure

```text
SignBridge/
│
├── metadata/
│   ├── jordanian_it/
│   ├── karsl/
│   ├── isharah/
│   └── unified/
│
├── models/
│
├── motion_library/
│   └── jordanian_it/
│
├── preprocessing/
│
├── training/
│
└── signbridge_integration/
    ├── backend/
    ├── frontend/
    ├── data_manifests/
    ├── docs/
    └── tools/
```

---

# Recognition Models

## Main Integrated Model

The primary recognition checkpoint used by the current SignBridge backend is:

```text
models/best_unified_karsl_multitask.pt
```

This checkpoint represents the final integrated multi-task recognition stage used in the current system.

Additional trained checkpoints are included for reproducibility and research traceability.

```text
models/
├── best_isharah_ctc.pt
├── best_karsl_isolated_model.pt
├── best_transfer_model.pt
├── best_unified_ctc.pt
├── best_unified_karsl_multitask.pt
├── hand_landmarker.task
└── holistic_landmarker.task
```

These checkpoints correspond to different stages of the SignBridge experimental pipeline, including:

- Isharah continuous recognition
- KArSL isolated-sign pretraining
- transfer-learning experiments
- unified CTC training
- final KArSL-integrated multi-task training

The included MediaPipe task files allow the recognition pipeline to use the landmark-extraction resources associated with the current project implementation.

---

# Training

Training scripts are available under:

```text
training/
```

Included scripts:

```text
train_isharah_ctc.py
train_karsl_isolated_pretrain.py
train_unified_sign_ctc.py
train_unified_karsl_multitask.py
finetune_jordanian_it.py
```

These scripts correspond to the major recognition-training stages conducted during the project.

---

# Preprocessing

Dataset preparation and feature-generation scripts are available under:

```text
preprocessing/
```

The preprocessing pipeline contains utilities for:

- signer-independent split generation
- Isharah pose preparation
- Isharah gloss extraction
- KArSL shared-feature preparation
- unified feature preparation
- unified dataset construction
- Jordanian IT dataset preparation

The repository therefore preserves the preprocessing workflow used to construct the recognition inputs and dataset configurations reported in the project.

---

# Metadata

Research metadata is stored under:

```text
metadata/
```

The metadata includes resources associated with:

- Jordanian IT
- KArSL
- Isharah
- unified multi-source training

Depending on the dataset and experiment, these files include:

- dataset splits
- vocabularies
- label mappings
- sequence metadata
- preparation summaries
- training histories
- audit reports
- model-preparation metadata
- signer-independent split information

---

# Datasets

Large raw datasets are intentionally not stored directly in the GitHub repository.

The repository instead contains the metadata, preparation scripts, trained checkpoints, and selected processed assets required to document and reproduce the SignBridge research workflow.

---

## Jordanian IT Sign Dataset

The SignBridge project includes a locally collected, domain-specific sign dataset containing IT-related signs used in the research prototype.

The dataset contains sign recordings associated with technical concepts relevant to the targeted educational setting.

The GitHub repository includes:

- dataset metadata
- signer-independent splits
- label mappings
- preprocessing configuration
- sequence metadata
- dataset audit files
- processed Jordanian IT motion files

The raw no-audio sign videos are hosted externally through Google Drive.

### Download

**Jordanian IT Dataset:**

https://drive.google.com/drive/folders/1WDEp-RBIxSrhKhOXuaoVrIeHnxt-ux9b?usp=sharing

Recommended local dataset location:

```text
data/jordanian_it/
```

> The locally collected dataset is research-specific and should not be interpreted as representing the complete vocabulary or linguistic variation of Jordanian Sign Language.

---

## KArSL

**KArSL** is an Arabic isolated-sign dataset used during the SignBridge recognition experiments.

Official dataset page:

https://hamzah-luqman.github.io/KArSL/

Official GitHub repository:

https://github.com/Hamzah-Luqman/KArSL

Recommended local location:

```text
data/external/karsl/
```

Raw KArSL files are **not redistributed** through this repository and remain subject to the original dataset's licensing and usage conditions.

---

## Isharah

**Isharah** is a continuous sign-language dataset used for continuous sign-recognition experiments in SignBridge.

Official project page:

https://snalyami.github.io/Isharah_CSLR/

Official GitHub repository:

https://github.com/snalyami/Isharah_CSLR

Recommended local location:

```text
data/external/isharah/
```

Raw Isharah files are **not redistributed** through this repository and remain subject to the original dataset's licensing and usage conditions.

---

# Motion Library

The repository includes the processed Jordanian IT motion library used by the web prototype:

```text
motion_library/jordanian_it/
```

These files provide locally stored motion assets that can be resolved by the SignBridge signed-output system during avatar playback.

Large motion collections derived from external datasets are not redistributed through this repository.

---

# Signed Output and Avatar Pipeline

The current SignBridge prototype separates sign recognition from signed-output generation.

The signed-output workflow follows the general structure:

```text
Educational Response
        ↓
Semantic Sign Planner
        ↓
Canonical Sign Tokens
        ↓
Motion Library Lookup
        ↓
Prepared Motion Assets
        ↓
Three.js Avatar Playback
```

Motion assets used by the current prototype are prepared offline and stored locally.

The avatar component is a **research prototype** and should not be interpreted as a linguistically validated system for unrestricted Jordanian Sign Language generation.

---

# Backend Setup

The backend is implemented using **FastAPI**.

### Recommended Python Version

```text
Python 3.13.x
```

Open PowerShell and navigate to the backend:

```powershell
cd signbridge_integration/backend
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

Create the local environment configuration from the provided example:

```powershell
Copy-Item ..\.env.example .env
```

Add the required API credentials to:

```text
.env
```

> Never commit the `.env` file or API credentials to GitHub.

Start the backend:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The backend should then be available at:

```text
http://127.0.0.1:8000
```

---

# Frontend Setup

The frontend is implemented using **Vite** and **Three.js**.

Open a second terminal and navigate to the frontend:

```powershell
cd signbridge_integration/frontend
```

Install frontend dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

Vite will display the local development URL in the terminal.

Open that URL in a browser to access the SignBridge web interface.

---

# Environment Variables

The SignBridge integration uses environment variables for API credentials and runtime configuration.

The example environment file is located at:

```text
signbridge_integration/.env.example
```

Create a local `.env` file before running services that require external API access.

Sensitive environment files are excluded from version control.

---

# Backend Python Dependencies

The backend dependency file is located at:

```text
signbridge_integration/backend/requirements.txt
```

Core dependencies include packages used for:

- numerical processing
- deep learning
- computer vision
- tabular metadata
- landmark extraction
- backend API services
- retrieval and educational generation

Important package names include:

```text
numpy
torch
opencv-python
pandas
mediapipe
```

> Python package names must match their official `pip` package names. For example, use `opencv-python` rather than `cv2`, and `torch` rather than `pytorch`.

---

# Avatar and Motion Assets

The frontend contains the avatar and motion assets required by the current SignBridge prototype.

Some avatar, character, motion-capture, or motion-generation resources originate from third-party tools or services.

Their original:

- licenses
- terms of use
- attribution requirements
- redistribution conditions

remain applicable.

Their presence in this research repository does not override the rights or licensing terms of the original providers.

---

# Reproducing the Project

A general reproduction workflow is:

1. Clone this repository.
2. Download the required datasets from their official or project-provided sources.
3. Place the datasets in the documented local directories.
4. Create and activate a Python virtual environment.
5. Install the backend Python dependencies.
6. Install the frontend dependencies.
7. Configure the required environment variables.
8. Run the relevant preprocessing scripts when reproducing dataset preparation.
9. Run the corresponding training scripts when reproducing model training.
10. Start the FastAPI backend.
11. Start the Vite frontend.
12. Open the SignBridge web interface.

---

# Files Intentionally Excluded from GitHub

Large or machine-specific files are intentionally excluded from version control where appropriate.

These may include:

- large raw external datasets
- extracted image/frame collections
- generated feature arrays
- temporary processing outputs
- caches
- Python virtual environments
- frontend `node_modules`
- frontend build output
- local environment files
- API credentials
- temporary debugging files

This keeps the repository portable while preserving the code, metadata, trained checkpoints, and research artifacts required to document the SignBridge workflow.

---

# Research Scope

The current SignBridge research prototype investigates the integration of:

- sign language recognition
- Jordanian IT-domain sign data
- continuous sign recognition
- isolated sign recognition
- signer-independent evaluation
- multi-source training
- multi-task learning
- transfer learning
- knowledge distillation
- landmark-based visual representation
- retrieval-augmented educational assistance
- educational content simplification
- semantic sign planning
- web-based interaction
- avatar-based sign presentation

SignBridge should be interpreted as a **research and educational prototype**, not as a complete or linguistically validated system for unrestricted Jordanian Sign Language recognition or translation.

---

# Important Notes

- Raw external datasets remain subject to their original licenses and usage conditions.
- KArSL and Isharah are not redistributed through this repository.
- The locally collected Jordanian IT dataset is research-specific and does not represent the complete Jordanian Sign Language vocabulary.
- API keys and private credentials must never be committed.
- Sensitive `.env` files are excluded from version control.
- Third-party avatar and motion assets remain subject to their original licenses and terms.
- Recognition vocabulary coverage does not necessarily imply equivalent validated avatar-motion coverage.
- The avatar and signed-output pipeline remain prototype components requiring further linguistic and Deaf-user evaluation.
- The included metadata, preprocessing scripts, training scripts, checkpoints, and motion assets correspond to the SignBridge research and integration workflow.

---

# Project

**SignBridge – SigmaX Research Project**