# Diabetes Prediction ML

A machine learning web application that predicts whether a patient is **Diabetic** or **Non-Diabetic** based on health indicators. Built with FastAPI and scikit-learn.

---

## Features

- Web form for entering patient health data
- Automatically trains and selects the best-performing ML model
- Returns a real-time prediction via a clean UI

---

## Tech Stack

| Layer | Tools |
|---|---|
| Web Framework | FastAPI, Jinja2 |
| ML Libraries | scikit-learn, XGBoost, CatBoost |
| Data Processing | pandas, NumPy |
| Language | Python 3 |

---

## Project Structure

```
diabetes-prediction-ml/
├── app.py                  # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── data/                   # Raw and cleaned datasets (CSV)
├── notebooks/              # Jupyter notebooks (EDA & model training)
├── templates/              # HTML templates (index.html)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Loads and splits dataset
│   │   ├── data_transformation.py  # Feature preprocessing pipeline
│   │   └── model_trainer.py        # Trains and selects best model
│   ├── pipeline/
│   │   └── predict_pipeline.py     # Loads model and runs predictions
│   ├── config.py           # App settings (paths, seed, env)
│   ├── exceptions.py       # Custom exception handling
│   ├── logger.py           # Logging setup
│   └── utils.py            # Helper functions (save/load objects)
└── artifacts/              # Generated after training (model.pkl, preprocessor.pkl)
```

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female / Other |
| Age | Numeric | Patient age (0–120) |
| Hypertension | Binary | 0 = No, 1 = Yes |
| Heart Disease | Binary | 0 = No, 1 = Yes |
| Smoking History | Categorical | Never / Former / Current |
| BMI | Numeric | Body Mass Index |
| HbA1c Level | Numeric | Average blood sugar over 3 months |
| Blood Glucose Level | Numeric | Current blood glucose (mg/dL) |

---

## ML Models Evaluated

The training pipeline evaluates the following classifiers and picks the best one based on accuracy:

- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/components/data_ingestion.py
```

This reads the cleaned dataset, transforms features, trains all models, and saves the best model and preprocessor to the `artifacts/` directory.

### 3. Run the web application

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser, fill in the health form, and get a prediction.

---

## Dataset

The dataset (`data/diabetes_prediction_dataset.csv`) contains patient health records with features like age, BMI, blood glucose levels, and medical history. A cleaned version (`diabetes_prediction_dataset_cleaned.csv`) is used for training.

---

## Output

The app returns one of:

- ✅ **Non-Diabetic**
- ⚠️ **Diabetic**
