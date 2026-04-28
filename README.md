# Data-Mining

# 🚬 Smoker Status Prediction System (Production-Level ML Project)

## 📌 Overview

This project builds a **production-ready machine learning system** to predict smoker status using bio-signal data. It goes beyond model training by implementing a **complete ML pipeline, API service, and deployment-ready architecture**.

The system leverages structured health data to classify individuals as smokers or non-smokers, combining **data preprocessing, feature engineering, model selection, and real-time inference via FastAPI**.

---

## 🧠 Key Features

* ✅ End-to-end ML pipeline (data → model → API)
* ✅ Multiple model comparison with cross-validation
* ✅ Best model selection based on F1-score
* ✅ Production-ready FastAPI service
* ✅ Named feature input (industry-standard API design)
* ✅ Feature validation & error handling
* ✅ Model persistence using `joblib`
* ✅ Clean, modular project structure

---

## ⚙️ ML Pipeline

### 1️⃣ Data Preprocessing

* Merged train & test datasets
* Removed duplicates and missing values
* Outlier detection using IQR method
* Standardized feature names for production usage
* Feature scaling using `StandardScaler` (within pipeline)

---

### 2️⃣ Feature Engineering

* Removed highly correlated features
* Encoded target variable using `LabelEncoder`
* Preserved feature schema for API consistency

---

### 3️⃣ Model Training & Evaluation

Trained and evaluated multiple models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree

Evaluation method:

* 5-fold Cross Validation
* Metrics: Accuracy, Precision, Recall, F1-score

---

### 📊 Model Performance

| Model             | Accuracy | Precision | Recall | F1-score   |
| ----------------- | -------- | --------- | ------ | ---------- |
| Gradient Boosting | 0.7578   | 0.7077    | 0.6788 | **0.6887** |
| SVM               | 0.7587   | 0.7112    | 0.6668 | 0.6793     |
| Random Forest     | 0.7500   | 0.6975    | 0.6593 | 0.6704     |

🏆 **Best Model:** Gradient Boosting

---

## 🚀 FastAPI Inference Service

The trained model is deployed using **FastAPI** for real-time predictions.

### ▶️ Run API

```bash
uvicorn src.api.app:app --reload
```

### 🌐 API Docs

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Usage

### 🔹 Endpoint: `/predict`

### ✅ Example Request

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

### ✅ Response

```json
{
  "prediction": "smoker"
}
```

---

## 🔐 Production Highlights

* ✔ Feature schema validation (prevents input errors)
* ✔ Consistent feature ordering using `features.pkl`
* ✔ End-to-end pipeline (no data leakage)
* ✔ Logging for training and API requests
* ✔ Scalable architecture for deployment

---

## 🔍 Future Improvements

* 🔹 Hyperparameter tuning (GridSearch / Optuna)
* 🔹 Deep learning models (PyTorch / TensorFlow)
* 🔹 Model monitoring & drift detection
* 🔹 Docker containerization
* 🔹 Cloud deployment (AWS / Render / Azure)
* 🔹 CI/CD pipeline integration

---

## 🎓 Acknowledgments

* University of Jaffna
* Kaggle Dataset: Smoker Status Prediction using Bio-Signals

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
