# Data-Mining

# 🚬 Smoker Status Prediction — Production ML System

## 📌 Overview

This project builds a **production-ready machine learning system** to predict smoker status using bio-signal data.
It goes beyond a typical ML notebook by implementing a **complete pipeline**, **model comparison**, and a **deployable FastAPI service**.

The goal is to demonstrate **end-to-end ML engineering skills** — from data preprocessing to API deployment.

---

## 🧠 Problem Statement

Given a set of physiological and medical features (bio-signals), predict whether a person is a **smoker or non-smoker**.

This is a **binary classification problem** with real-world applications in:

* Healthcare analytics
* Risk assessment systems
* Preventive medical diagnostics

---

## ⚙️ Tech Stack

**Languages & Libraries**

* Python
* Pandas, NumPy
* Scikit-learn

**ML & Engineering**

* Pipeline (ColumnTransformer + StandardScaler)
* Cross-validation
* Feature engineering (outlier removal, correlation filtering)

**Backend**

* FastAPI
* Pydantic (schema validation)

**Tools**

* Git & GitHub
* Uvicorn

---

## 🚀 Key Features

✔ End-to-end ML pipeline
✔ Cleaned & standardized feature engineering
✔ Correlation-based feature selection
✔ Multiple model training & comparison
✔ Cross-validation (5-fold)
✔ Production-ready FastAPI API
✔ Named feature input (industry standard)
✔ Robust input validation
✔ Reproducible model artifacts

---

## 📊 Dataset

* Source: Kaggle
* Type: Bio-signal dataset
* Features include:

  * Age, height, weight
  * Blood pressure (systolic, relaxation)
  * Cholesterol, glucose, triglycerides
  * Liver enzymes (AST, ALT, GTP)
  * Dental and hearing metrics

---

## 🧪 ML Pipeline

### 1. Data Preprocessing

* Removed duplicates and null values
* Outlier removal using IQR method
* Standardized numerical features

### 2. Feature Engineering

* Cleaned feature names (API-safe format)
* Removed highly correlated features (>0.8 threshold)

### 3. Model Training

Trained multiple models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 📈 Model Performance

| Model             | Accuracy | Precision | Recall | F1 Score  |
| ----------------- | -------- | --------- | ------ | --------- |
| Gradient Boosting | ~0.75    | ~0.70     | ~0.67  | **~0.69** |
| SVM               | ~0.75    | ~0.71     | ~0.66  | ~0.68     |
| Random Forest     | ~0.75    | ~0.69     | ~0.65  | ~0.67     |

🏆 **Best Model: Gradient Boosting** (balanced performance)


---

## 🔌 API Usage

### Run Locally

```bash
pip install -r requirements.txt
python train.py
uvicorn src.api.app:app --reload
```

---

### Endpoint

**POST** `/predict`

---

### Input (Named Features)

```json
{
  "data": {
    "age": 45,
    "heightcm": 170,
    "weightkg": 70,
    "waistcm": 80,
    "eyesight_left": 1.0,
    "eyesight_right": 1.0,
    "hearing_left": 1,
    "hearing_right": 1,
    "systolic": 120,
    "relaxation": 80,
    "fasting_blood_sugar": 90,
    "cholesterol": 180,
    "triglyceride": 150,
    "hdl": 50,
    "hemoglobin": 14,
    "urine_protein": 1,
    "serum_creatinine": 1.0,
    "ast": 20,
    "alt": 25,
    "gtp": 30,
    "dental_caries": 0
  }
}
```

---

### Output

```json
{
  "prediction": "smoker"
}
```

---

## 🧠 Key Learnings

* Importance of **pipeline-based preprocessing**
* Avoiding **data leakage in ML systems**
* Designing **robust and scalable APIs**
* Handling **feature consistency in production**
* Building **end-to-end ML workflows**

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearch / Optuna)
* Deep learning models
* Model monitoring & logging
* Deployment (Render / AWS / Docker)
* MLflow integration

---

## 👤 Author

**Ravixshana Atputharavi**
BSc Eng (Hons) in Computer Engineering
University of Jaffna

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
