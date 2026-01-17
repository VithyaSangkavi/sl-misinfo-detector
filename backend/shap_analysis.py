import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

# Paths
DATA_PATH = os.path.join("data", "sl_misinfo_headlines_medium_srilanka.csv")
MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CAT_MODEL_PATH = os.path.join(MODELS_DIR, "catboost_model.cbm")

LABEL_MAP = {
    0: "fake",
    1: "misleading",
    2: "satire",
    3: "real",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df["headline"].astype(str)
    y = df["label"].astype(int)
    return df, X, y


def main():
    print("Loading data, vectorizer, and CatBoost model...")
    df, X, y = load_data()

    vectorizer = joblib.load(VEC_PATH)
    cat_model = CatBoostClassifier()
    cat_model.load_model(CAT_MODEL_PATH)

    # For SHAP we don't want all 40k rows (too slow) – sample e.g. 1500
    sample_size = min(1500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X_sample = df_sample["headline"].astype(str)
    y_sample = df_sample["label"].astype(int)

    # Vectorise and convert to dense for CatBoost/SHAP
    X_sample_vec = vectorizer.transform(X_sample)
    X_sample_dense = X_sample_vec.toarray()

    feature_names = vectorizer.get_feature_names_out()

    print("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_sample_dense)
    # shap_values: list of arrays, length = #effective_classes (K or K-1)
    n_shap_classes = len(shap_values)
    has_baseline = (n_shap_classes == len(LABEL_MAP) - 1)

    os.makedirs("reports", exist_ok=True)

    # ---------- GLOBAL FEATURE IMPORTANCE ----------
    print("Computing global feature importance...")
    # Stack shap_values to shape (n_samples, n_features, n_classes_eff)
    sv_stack = np.stack(shap_values, axis=-1)
    # Mean absolute SHAP across samples and effective classes
    mean_abs_shap = np.mean(np.abs(sv_stack), axis=(0, 2))
    # Sort features by importance
    idx_sorted = np.argsort(mean_abs_shap)[::-1]
    top_n = 30
    top_idx = idx_sorted[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = mean_abs_shap[top_idx]

    plt.figure(figsize=(8, 10))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1])
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel("Mean |SHAP value| (global importance)")
    plt.title("Top TF-IDF features influencing CatBoost predictions")
    plt.tight_layout()
    bar_path = os.path.join("reports", "shap_global_bar_top30.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved global SHAP bar plot to {bar_path}")

    # ---------- GLOBAL FAKE SHAP SUMMARY ----------
    # Case 1: SHAP returned K-1 arrays → fake is baseline → reconstruct
    # Case 2: SHAP returned K arrays → fake is index 0
    if has_baseline:
        print("Reconstructing SHAP values for baseline class 'fake' (global)...")
        # shap_values is list length 3, each (n_samples, n_features)
        shap_fake_global = np.zeros_like(shap_values[0])
        for c in range(n_shap_classes):
            shap_fake_global -= shap_values[c]

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_fake_global,
            feature_names=feature_names,
            show=False,
            max_display=25,
        )
        fake_beeswarm_path = os.path.join(
            "reports", "shap_summary_fake_reconstructed_beeswarm.png"
        )
        plt.tight_layout()
        plt.savefig(fake_beeswarm_path, dpi=300)
        plt.close()
        print(
            f"Saved reconstructed SHAP summary (beeswarm) plot for 'fake' to {fake_beeswarm_path}"
        )
    else:
        print("Creating SHAP summary plot for class 'fake' (index 0)...")
        shap_values_fake = shap_values[0]  # index 0 = fake

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values_fake,
            feature_names=feature_names,
            show=False,
            max_display=25,
        )
        fake_beeswarm_path = os.path.join("reports", "shap_summary_fake_beeswarm.png")
        plt.tight_layout()
        plt.savefig(fake_beeswarm_path, dpi=300)
        plt.close()
        print(f"Saved SHAP summary (beeswarm) plot for 'fake' to {fake_beeswarm_path}")

    # ---------- LOCAL EXPLANATIONS (EXAMPLES) ----------
    print("Saving per-example SHAP bar plots...")
    example_count = 3
    df_examples = df_sample.sample(n=example_count, random_state=123).reset_index(
        drop=True
    )
    X_ex = df_examples["headline"].astype(str)
    X_ex_vec = vectorizer.transform(X_ex).toarray()

    shap_values_ex = explainer.shap_values(X_ex_vec)
    n_shap_classes_ex = len(shap_values_ex)
    preds = cat_model.predict(X_ex_vec).reshape(-1)

    for i in range(example_count):
        text = X_ex.iloc[i]
        label_id = int(preds[i])
        label_name = LABEL_MAP.get(label_id, str(label_id))

        # Predicted class index in SHAP arrays:
        # - If SHAP gives all 4 classes, index = label_id
        # - If SHAP gives 3 classes (baseline missing), we clamp to last index as fallback
        pred_class_index = min(label_id, n_shap_classes_ex - 1)
        shap_vals_for_pred = shap_values_ex[pred_class_index][i]

        # --- Predicted class SHAP bar plot ---
        contrib_idx = np.argsort(np.abs(shap_vals_for_pred))[::-1][:10]
        contrib_feats = [feature_names[j] for j in contrib_idx]
        contrib_vals = shap_vals_for_pred[contrib_idx]

        plt.figure(figsize=(7, 4))
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in contrib_vals]
        y_pos = np.arange(len(contrib_feats))
        plt.barh(y_pos, contrib_vals[::-1], color=colors[::-1])
        plt.yticks(y_pos, contrib_feats[::-1])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"SHAP contributions for predicted class '{label_name}'")
        plt.xlabel("SHAP value (impact on model output)")
        plt.tight_layout()

        filename = f"shap_example_{i+1}_{label_name}.png"
        ex_path = os.path.join("reports", filename)
        plt.savefig(ex_path, dpi=300)
        plt.close()

        print(f"Example {i+1}:")
        print(f"Headline: {text}")
        print(f"Predicted label: {label_name} (id={label_id})")
        print(f"Saved SHAP example plot to {ex_path}")

        # --- ALSO: per-example SHAP for 'fake' ---
        if has_baseline and n_shap_classes_ex == len(LABEL_MAP) - 1:
            # fake is baseline → reconstruct for this sample
            shap_fake_i = np.zeros_like(shap_values_ex[0][i])
            for c in range(n_shap_classes_ex):
                shap_fake_i -= shap_values_ex[c][i]
        else:
            # fake is explicit class at index 0
            shap_fake_i = shap_values_ex[0][i]

        fake_contrib_idx = np.argsort(np.abs(shap_fake_i))[::-1][:10]
        fake_contrib_feats = [feature_names[j] for j in fake_contrib_idx]
        fake_contrib_vals = shap_fake_i[fake_contrib_idx]

        plt.figure(figsize=(7, 4))
        fake_colors = ["#22c55e" if v > 0 else "#ef4444" for v in fake_contrib_vals]
        y_pos = np.arange(len(fake_contrib_feats))
        plt.barh(y_pos, fake_contrib_vals[::-1], color=fake_colors[::-1])
        plt.yticks(y_pos, fake_contrib_feats[::-1])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("SHAP contributions for class 'fake'")
        plt.xlabel("SHAP value (impact on model output)")
        plt.tight_layout()

        fake_filename = f"shap_example_{i+1}_fake.png"
        fake_path = os.path.join("reports", fake_filename)
        plt.savefig(fake_path, dpi=300)
        plt.close()

        print(f"Also saved SHAP plot for 'fake' to {fake_path}\n")

    print("SHAP analysis done.")


if __name__ == "__main__":
    main()
