import os
import joblib
import numpy as np

from typing import List, Dict


MODELS_DIR = "models"

# Adjust this to match the actual filename printed in train_models.py
BEST_MODEL_FILENAME = None  # we’ll detect it

LABEL_MAP = {
    0: "fake",
    1: "misleading",
    2: "satire",
    3: "real"
}


def load_vectorizer_and_model():
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    vectorizer = joblib.load(vec_path)

    # auto-pick the best model file
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith("best_model_")]
    if not files:
        raise RuntimeError("No best_model_*.joblib found in models/")

    model_path = os.path.join(MODELS_DIR, files[0])
    model = joblib.load(model_path)

    # crude detection if it's CatBoost
    is_catboost = "catboost" in files[0].lower()

    return vectorizer, model, is_catboost


VECTORIZER, MODEL, IS_CATBOOST = load_vectorizer_and_model()


def predict_headlines(headlines: List[str]) -> List[Dict]:
    X_vec = VECTORIZER.transform(headlines)
    if IS_CATBOOST:
        X_inp = X_vec.toarray()
        preds = MODEL.predict(X_inp)
        # CatBoost returns shape (n,1) or (n,) – normalize
        preds = preds.reshape(-1)
        proba = MODEL.predict_proba(X_inp)
    else:
        X_inp = X_vec
        preds = MODEL.predict(X_inp)
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X_inp)
        else:
            # SVM without probability: fake a distribution via decision_function
            df = MODEL.decision_function(X_inp)
            # convert to pseudo-probabilities
            exp = np.exp(df - df.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)

    results = []
    for i, text in enumerate(headlines):
        label_id = int(preds[i])
        label_name = LABEL_MAP.get(label_id, str(label_id))
        probs = {LABEL_MAP[j]: float(proba[i, j]) for j in range(len(LABEL_MAP))}
        results.append({
            "headline": text,
            "label_id": label_id,
            "label_name": label_name,
            "probabilities": probs
        })
    return results
