# Lifestyle & Age at Death – ML Project

=====================================

**Course**: M. Grum: Advanced AI-based Application Systems  
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam  
**Repository**: Forked from [MarcusGrum/AI-CPS](https://github.com/MarcusGrum/AI-CPS)

## Ownership

**Project Team**: 

- Student Name 1: Moaz Hussein - GitHub: MoazEmad1
- Student Name 2: Nathan Fernandes - GitHub: Alpha-Strikes

This repository is part of the course 'M. Grum: Advanced AI-based Application Systems' and follows the project structure requirements from the AI-CPS repository.

## Project Overview

This project predicts age at death based on lifestyle factors (work hours, rest hours, sleep hours, exercise hours) and occupation type. The Data for this project is scrapped from kaggle https://www.kaggle.com/datasets/oluwatosinadewale/quality-of-life-data. It implements:

- **OLS Regression** (Statsmodels) for baseline comparison
- **TensorFlow/Keras ANN** for non-linear modeling
- **Data scraping** from a GitHub repository
- **Docker containerization** for model deployment

## Project Structure

Current structure (aligned with AI-CPS repository):

- `data/` – cleaned / intermediate datasets
- `code/` – all Python code files:
  - `data_prep.py` – loading, cleaning, train/test split helpers
  - `eda.py` – exploratory data analysis and visualizations
  - `ols_model.py` – OLS regression training & evaluation
  - `ann_model.py` – TensorFlow/Keras ANN regressor for age-at-death
  - `annRequests_PyBrain/` – PyBrain ANN request handlers (from AI-CPS)
  - `annRequests_TensorFlow/` – TensorFlow ANN request handlers (from AI-CPS)
  - `experiments/` – experiment scripts (from AI-CPS)
  - `messageClient/` – MQTT communication client (from AI-CPS)
- `figures/` – saved plots (diagnostic plots, scatter plots, training curves)
- `outputs/` – metrics, logs, and model summaries
- `models/` – saved models (`currentOlsSolution.pkl`, `currentAiSolution.h5`)
- `learningBase/` – training/validation performance metrics and visualizations
- `data/` – processed datasets:
  - `joint_data_collection.csv` – cleaned and normalized dataset
  - `training_data.csv` – 80% of data for training
  - `test_data.csv` – 20% of data for testing
  - `activation_data.csv` – single entry for model activation
- `requirements.txt` – Python dependencies
- `Dockerfile` – container definitions (for learningBase, activationBase, knowledgeBase, codeBase)

## How to Run

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data (Subgoal 2)

```bash
cd code
python data_prep.py
```

This creates:

- `data/joint_data_collection.csv` – cleaned and normalized dataset
- `data/training_data.csv` – 80% for training
- `data/test_data.csv` – 20% for testing
- `data/activation_data.csv` – 1 entry for activation

### 3. Run OLS Model (Subgoal 5)

```bash
cd code
python ols_model.py
```

This:

- Trains OLS model using Statsmodels
- Generates diagnostic plots and scatter plots
- Saves model as `models/currentOlsSolution.pkl`
- Saves performance metrics to `learningBase/`

### 4. Run ANN Model (Subgoal 4)

```bash
cd code
python ann_model.py
```

This:

- Trains TensorFlow/Keras ANN model
- Generates training/testing curves, diagnostic plots, scatter plots
- Saves model as `models/currentAiSolution.h5`
- Saves performance metrics to `learningBase/`

## License

This project is committed to the **AGPL-3.0 license** as required by the course.

## Docker Images

The project includes Docker images for model deployment:

- `learningBase_yourIndividualProjektTitle` – training and test data
- `activationBase_yourIndividualProjektTitle` – activation data
- `knowledgeBase_yourIndividualProjektTitle` – trained models
- `codeBase_yourIndividualProjektTitle` – model code

See `PDF_REQUIREMENTS_CHECKLIST.md` for detailed progress tracking.
