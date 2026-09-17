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



Dataset preparation and feature extraction scripts are available under:



```text

preprocessing/

```



\## Metadata



Dataset splits, vocabularies, preparation summaries, training histories, and audit files are stored under:



```text

metadata/

```



\## Datasets



Large raw datasets are intentionally not stored directly in this GitHub repository.



\### Jordanian IT Sign Dataset



The project includes a domain-specific Jordanian Sign Language dataset containing IT-related signs.



The repository includes:



\- dataset metadata

\- signer-independent splits

\- label mappings

\- preprocessing configuration

\- sequence metadata

\- processed Jordanian IT motion files



Raw videos are hosted externally.



Jordanian IT dataset link:



```text

ADD\_GOOGLE\_DRIVE\_LINK\_HERE

```



Expected local location:



```text

data/jordanian\_it/

```



\### KArSL



Official source:



https://github.com/Hamzah-Luqman/KArSL/blob/main/index.html



Expected local location:



```text

data/external/karsl/

```



Raw KArSL files are not redistributed in this repository.



\### Isharah



Official source:



https://github.com/snalyami/Isharah\_CSLR



Expected local location:



```text

data/external/isharah/

```



Raw Isharah files are not redistributed in this repository.



\## Motion Library



The repository includes the processed Jordanian IT motion library:



```text

motion\_library/jordanian\_it/

```



Large externally derived KArSL and Isharah motion libraries are not included.



\## Backend Setup



```powershell

cd signbridge\_integration/backend

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

Copy-Item ..\\.env.example .env

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```



Add the required API credentials to `.env`.



Do not commit `.env`.



\## Frontend Setup



```powershell

cd signbridge\_integration/frontend

npm install

npm run dev

```



Open the local URL shown by Vite.



\## Reproducibility Notes



To reproduce the full pipeline:



1\. Clone this repository.

2\. Download the required datasets.

3\. Place them in the documented dataset directories.

4\. Install the required dependencies.

5\. Run the preprocessing scripts.

6\. Run the corresponding training scripts.



Large raw videos, generated feature arrays, caches, and temporary files are excluded from GitHub.



\## Research Scope



SignBridge combines:



\- sign language recognition

\- Jordanian IT-domain sign data

\- continuous sign recognition

\- multi-source training

\- retrieval-augmented educational assistance

\- simplified educational explanations

\- web-based interaction

\- avatar-based sign presentation



\## Important Notes



\- Raw external datasets remain subject to their original licenses and usage conditions.

\- API keys and private credentials must never be committed.

\- Third-party avatar and motion assets remain subject to their original licenses and terms.



\## Project



SignBridge – SigmaX Research Project

