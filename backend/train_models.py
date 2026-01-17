import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from catboost import CatBoostClassifier


DATA_PATH = os.path.join("data", "sl_misinfo_headlines_medium_srilanka.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    # We use headline as X, label as y (0=fake, 1=misleading, 2=satire, 3=real)
    X = df["headline"].astype(str)
    y = df["label"].astype(int)
    return X, y, df


def vectorize_text(X_train, X_val, X_test):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_val_vec, X_test_vec


def evaluate_model(name, model, X_val_vec, y_val):
    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"\n {name} ")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(classification_report(y_val, y_pred, digits=4))
    return acc, f1


def main():
    print("Loading data...")
    X, y, df = load_data()
    print(f"Dataset size: {len(df)} rows")

    # Train/val/test split: 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,   # 0.25 of 0.8 = 0.2 overall
        random_state=42,
        stratify=y_temp
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer, X_train_vec, X_val_vec, X_test_vec = vectorize_text(
        X_train, X_val, X_test
    )

    # Save vectorizer
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"Saved TF-IDF vectorizer to {vec_path}")

    results = []

    # 1) Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000
    )
    log_reg.fit(X_train_vec, y_train)
    acc, f1 = evaluate_model("Logistic Regression", log_reg, X_val_vec, y_val)
    results.append(("log_reg", log_reg, acc, f1))

    # 2) Linear SVM
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    acc, f1 = evaluate_model("Linear SVM", svm, X_val_vec, y_val)
    results.append(("svm", svm, acc, f1))

    # 3) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_vec, y_train)
    acc, f1 = evaluate_model("Random Forest", rf, X_val_vec, y_val)
    results.append(("rf", rf, acc, f1))

    # 4) CatBoost (new algorithm for assignment)
    # Need dense input: convert from sparse -> dense (might use subset if RAM is low)
    X_train_dense = X_train_vec.toarray()
    X_val_dense = X_val_vec.toarray()

    cat_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        verbose=False,
        random_seed=42
    )
    cat_model.fit(X_train_dense, y_train)
    acc, f1 = evaluate_model("CatBoost", cat_model, X_val_dense, y_val)
    results.append(("catboost", cat_model, acc, f1))

    cat_model_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    cat_model.save_model(cat_model_path)
    print(f"Saved CatBoost model to {cat_model_path}")

    # Pick best by validation F1
    results.sort(key=lambda x: x[3], reverse=True)
    best_name, best_model, best_acc, best_f1 = results[0]

    print("\n BEST MODEL ON VALIDATION")
    print(f"Model: {best_name}")
    print(f"Val Accuracy: {best_acc:.4f}")
    print(f"Val F1: {best_f1:.4f}")

    # Evaluate best on test set
    if best_name == "catboost":
        X_test_for_best = X_test_vec.toarray()
    else:
        X_test_for_best = X_test_vec

    y_test_pred = best_model.predict(X_test_for_best)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("\n TEST PERFORMANCE (Best Model)")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Save best model
    model_path = os.path.join(MODELS_DIR, f"best_model_{best_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model ({best_name}) to {model_path}")


if __name__ == "__main__":
    main()
