# Data-Mining
 Binary prediction of smoker status bio-signals

## 📌 Introduction
This project focuses on **data preprocessing, feature selection, model training, and evaluation** for a classification problem. The dataset, sourced from Kaggle, consists of `train_dataset.csv` and `test_dataset.csv`. The primary objective is to compare multiple machine learning models and determine the most effective one for classification.

## 🚀 Workflow
### 1️⃣ Data Download and Preprocessing
- **Downloaded** `train_dataset.csv` and `test_dataset.csv` using Kaggle API.
- **Merged** train and test datasets for preprocessing.
- **Explored** dataset structure (`.info()`, `.head()`), checked for missing values and duplicates.
- **Cleaned** dataset by removing duplicates and handling null values.
- **Detected & removed outliers** using **Interquartile Range (IQR)**.
- **Standardized** numerical features using **StandardScaler** (important for models like SVM and Logistic Regression).

### 2️⃣ Feature Engineering
- **Correlation-based filtering** to remove highly correlated features.
- **Dropped unnecessary columns**.
- **Encoded categorical target variables** using **LabelEncoder**.

### 3️⃣ Model Training & Evaluation
- **Split dataset** into training and testing sets.
- **Trained multiple models:**
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (K-NN)
  - Naïve Bayes
  - Decision Tree
- **Performed Cross-Validation** (5-fold) to evaluate models.
- **Used scoring metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score

### 4️⃣ Model Performance & Visualization
- Compared models based on classification metrics.
- Visualized model performance using plots and graphs.

## 📊 Results & Best Model
| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Gradient Boosting | 0.753909 | 0.702541      | 0.675012    | 0.684548 |
| SVM                 | 0.754577   | 0.705123  | 0.663434  | 0.675290     |

**🏆 Gradient Boosting was the best-performing model**, balancing precision and recall effectively.

## 🔍 Future Improvements
- **Deep Learning Models:** Using neural networks to enhance prediction accuracy.
- **Feature Expansion:** Incorporating lifestyle, genetic, or environmental data.
- **Hyperparameter Tuning:** Optimizing models with Grid Search or Random Search.

## ⭐ Acknowledgments
- **University Of Jaffna**
- **Datasets from Kaggle**

If you find this project useful, **leave a star ⭐ on GitHub!**
