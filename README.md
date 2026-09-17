\# SignBridge



SignBridge is an AI-powered educational assistant designed to support Deaf students through sign language recognition, educational content retrieval, simplified explanations, and avatar-based sign presentation.



The project focuses primarily on Jordanian Sign Language and includes experiments using the Jordanian IT dataset, KArSL, and Isharah.



\## Repository Structure



```text

SignBridge/

├── metadata/

├── models/

├── motion\_library/

├── preprocessing/

├── training/

└── signbridge\_integration/

&#x20;   ├── backend/

&#x20;   ├── frontend/

&#x20;   ├── data\_manifests/

&#x20;   ├── docs/

&#x20;   └── tools/

```



\## Main Model



The main integrated model used by the current backend is:



```text

models/best\_unified\_karsl\_multitask.pt

```



Other trained checkpoints are also included under:



```text

models/

```



These include the intermediate Isharah, KArSL, transfer-learning, and unified checkpoints used during the project experiments.



\## Training Scripts



Training scripts are available under:



```text

training/

```



including:



\- `train\_isharah\_ctc.py`

\- `train\_karsl\_isolated\_pretrain.py`

\- `train\_unified\_sign\_ctc.py`

\- `train\_unified\_karsl\_multitask.py`

\- `finetune\_jordanian\_it.py`



\## Preprocessing



Dataset preparation, split generation, pose preparation, and shared-feature extraction scripts are available under:



```text

preprocessing/

```



The repository includes the preprocessing scripts required for the Jordanian IT, KArSL, Isharah, and unified training pipelines used in SignBridge.



\## Metadata



Dataset splits, vocabularies, preparation summaries, training histories, audit files, label mappings, and model-preparation metadata are stored under:



```text

metadata/

```



The metadata directory contains separate resources for:



\- Jordanian IT

\- KArSL

\- Isharah

\- Unified training



\## Datasets



Large raw datasets are intentionally not stored directly in this GitHub repository.



\### Jordanian IT Sign Dataset



The SignBridge project includes a domain-specific Jordanian Sign Language dataset containing IT-related signs collected for this research project.



The repository includes:



\- dataset metadata

\- signer-independent splits

\- label mappings

\- preprocessing configuration

\- sequence metadata

\- dataset audit files

\- processed Jordanian IT motion files



Raw videos are hosted externally.



Jordanian IT dataset download:



```text

ADD\_GOOGLE\_DRIVE\_LINK\_HERE

```



Expected local location after download:



```text

data/jordanian\_it/

```



\### KArSL



KArSL is an Arabic Sign Language isolated-sign dataset used in the SignBridge recognition experiments.



Official dataset page:



https://hamzah-luqman.github.io/KArSL/



Official GitHub repository:



https://github.com/Hamzah-Luqman/KArSL



Expected local location:



```text

data/external/karsl/

```



Raw KArSL files are not redistributed through this repository.



\### Isharah



Isharah is a continuous sign-language dataset used in the continuous sign-recognition experiments.



Official project page:



https://snalyami.github.io/Isharah\_CSLR/



Official GitHub repository:



https://github.com/snalyami/Isharah\_CSLR



Expected local location:



```text

data/external/isharah/

```



Raw Isharah files are not redistributed through this repository.



\## Motion Library



The repository includes the processed Jordanian IT motion library:



```text

motion\_library/jordanian\_it/

```



This contains the runtime motion files used for the Jordanian IT sign vocabulary.



Large externally derived KArSL and Isharah motion libraries are not included in the repository.



\## Models



The repository includes the trained checkpoints used during the project:



```text

models/

├── best\_isharah\_ctc.pt

├── best\_karsl\_isolated\_model.pt

├── best\_transfer\_model.pt

├── best\_unified\_ctc.pt

├── best\_unified\_karsl\_multitask.pt

├── hand\_landmarker.task

└── holistic\_landmarker.task

```



The MediaPipe task files are included so the recognition pipeline can use the same landmark-extraction resources as the current project setup.



\## Backend Setup



Open PowerShell and navigate to the backend:



```powershell

cd signbridge\_integration/backend

```



Create a virtual environment:



```powershell

python -m venv .venv

```



Activate it:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



Create the local environment file from the provided template:



```powershell

Copy-Item ..\\.env.example .env

```



Add the required API credentials to `.env`.



Do not commit `.env`.



Start the backend:



```powershell

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```



The backend should then be available at:



```text

http://127.0.0.1:8000

```



\## Frontend Setup



Open another terminal and navigate to the frontend:



```powershell

cd signbridge\_integration/frontend

```



Install frontend dependencies:



```powershell

npm install

```



Start the Vite development server:



```powershell

npm run dev

```



Open the local URL displayed by Vite in the terminal.



\## Environment Variables



The integration uses environment variables for configuration and API credentials.



The example environment file is located at:



```text

signbridge\_integration/.env.example

```



Sensitive `.env` files are excluded from version control.



\## Avatar and Motion Assets



The frontend contains the avatar and motion assets used by the SignBridge prototype.



Some avatar and motion-generation assets originate from third-party tools or services. Their original licenses, terms of use, and redistribution conditions remain applicable.



\## Reproducibility Notes



To reproduce the complete SignBridge pipeline:



1\. Clone this repository.

2\. Download the required datasets from their official or project-provided sources.

3\. Place the datasets in the documented local directories.

4\. Install the required Python dependencies.

5\. Install the frontend dependencies.

6\. Configure the required environment variables.

7\. Run the preprocessing scripts when reproducing dataset preparation.

8\. Run the corresponding training scripts when reproducing model training.

9\. Start the backend and frontend for the integrated SignBridge application.



Large raw videos, extracted frame collections, generated feature arrays, caches, temporary files, virtual environments, and frontend build dependencies are intentionally excluded from GitHub because of size and reproducibility considerations.



\## Research Scope



SignBridge combines:



\- sign language recognition

\- Jordanian IT-domain sign data

\- continuous sign recognition

\- isolated sign recognition

\- multi-source training

\- transfer learning

\- retrieval-augmented educational assistance

\- educational content simplification

\- web-based sign interaction

\- avatar-based sign presentation



\## Important Notes



\- Raw external datasets remain subject to their original licenses and usage conditions.

\- API keys and private credentials must never be committed.

\- Third-party avatar and motion assets remain subject to their original licenses and terms.

\- Large external datasets must be downloaded separately.

\- The included metadata, training scripts, preprocessing scripts, checkpoints, and Jordanian IT motion library correspond to the SignBridge research and integration workflow.



\## Project



SignBridge – SigmaX Research Project



