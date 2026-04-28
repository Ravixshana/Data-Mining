import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ================================
# 1. LOAD DATA
# ================================
def load_data():
    train = pd.read_csv("data/raw/train_dataset.csv")
    test = pd.read_csv("data/raw/test_dataset.csv")

    train["dataset"] = "train"
    test["dataset"] = "test"

    return pd.concat([train, test], ignore_index=True)


# ================================
# 2. CLEAN COLUMN NAMES
# ================================
def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


# ================================
# 3. CLEAN DATA
# ================================
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


# ================================
# 4. REMOVE OUTLIERS
# ================================
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col == "smoking":
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# ================================
# 5. REMOVE CORRELATED FEATURES
# ================================
def remove_high_corr(X, threshold=0.8):
    corr_matrix = X.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(abs(upper[col]) > threshold)]

    return X.drop(columns=to_drop), to_drop


# ================================
# 6. PREPARE DATA
# ================================
def prepare_data(df):
    df = df.drop(columns=["dataset"], errors="ignore")

    X = df.drop("smoking", axis=1)
    y = df["smoking"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    # 🔥 REMOVE CORRELATED FEATURES SAFELY
    X, dropped = remove_high_corr(X)

    print("Dropped correlated features:", dropped)

    return X, y, le


# ================================
# 7. PIPELINE
# ================================
def build_pipeline(model, numeric_features):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features)
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])


# ================================
# 8. MODELS
# ================================
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }


# ================================
# 9. TRAIN
# ================================
def train_models(X, y):
    numeric_features = X.select_dtypes(include=np.number).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()

    best_model = None
    best_score = 0
    results = []

    for name, model in models.items():

        pipeline = build_pipeline(model, numeric_features)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        )

        result = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision_macro"].mean(),
            "Recall": scores["test_recall_macro"].mean(),
            "F1": scores["test_f1_macro"].mean()
        }

        results.append(result)

        if result["F1"] > best_score:
            best_score = result["F1"]
            best_model = pipeline

    # 🔥 VERY IMPORTANT → FIT FINAL MODEL
    best_model.fit(X_train, y_train)

    return best_model, pd.DataFrame(results)


# ================================
# 10. MAIN
# ================================
def main():

    print("Loading data...")
    df = load_data()

    print("Cleaning columns...")
    df = clean_column_names(df)

    print("Cleaning data...")
    df = clean_data(df)

    print("Removing outliers...")
    df = remove_outliers(df)

    print("Preparing data...")
    X, y, le = prepare_data(df)

    print("Training...")
    best_model, results = train_models(X, y)

    print(results)

    # 🔥 SAVE EVERYTHING
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")

    print("✅ Training complete & artifacts saved!")


if __name__ == "__main__":
    main()