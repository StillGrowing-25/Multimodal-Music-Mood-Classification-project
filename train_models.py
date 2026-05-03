"""
Train and compare four classifiers for MIREX Mood Recognition.

Target label: cluster (5 classes: Cluster 1 - Cluster 5)
              (use --category flag to train on 28 subcategories instead)

Models:
  1. audio_only     -- SVM trained on audio features only
  2. text_only      -- Logistic Regression on TF-IDF lyrics features
  3. early_fusion   -- MLP trained on [audio || text] concatenation
  4. late_fusion    -- Weighted average of audio + text predicted probabilities

Saves:
  models/{name}.joblib        -- per-model bundle
  models/best_model.joblib    -- best model by dev accuracy
  reports/model_results.csv   -- comparison table

Run:
  python train_models.py              # train on cluster labels (5-class)
  python train_models.py --category   # train on subcategory labels (28-class)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

FEATURES_CSV = Path("data/features.csv")
MODELS_DIR   = Path("models")
RESULTS_CSV  = Path("reports/model_results.csv")
RANDOM_STATE = 42
TOP_K_AUDIO  = 200
TOP_K_TEXT   = 1000


# ── feature helpers ────────────────────────────────────────────────────────

def get_split(df, split, audio_cols, text_cols, le):
    mask   = df["split"] == split
    Xa     = df.loc[mask, audio_cols].values.astype(np.float32)
    Xt     = df.loc[mask, text_cols].values.astype(np.float32)
    Xb     = np.hstack([Xa, Xt])
    y      = le.transform(df.loc[mask, "label"].values)
    return Xa, Xt, Xb, y


# ── model builders ─────────────────────────────────────────────────────────

def build_audio_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=TOP_K_AUDIO)),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)),
    ])


def build_text_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("select", SelectKBest(f_classif, k=TOP_K_TEXT)),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def build_early_fusion_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=TOP_K_AUDIO + TOP_K_TEXT)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
            random_state=RANDOM_STATE,
        )),
    ])


# ── late fusion ────────────────────────────────────────────────────────────

class LateFusionClassifier:
    """Weighted average of predicted probabilities from two fitted models."""
    def __init__(self, audio_model, text_model, audio_weight=0.5, text_weight=0.5):
        self.audio_model  = audio_model
        self.text_model   = text_model
        self.audio_weight = audio_weight
        self.text_weight  = text_weight

    def predict_proba(self, Xa, Xt):
        return (self.audio_weight * self.audio_model.predict_proba(Xa) +
                self.text_weight  * self.text_model.predict_proba(Xt))

    def predict(self, Xa, Xt):
        return np.argmax(self.predict_proba(Xa, Xt), axis=1)

    def tune_weights(self, Xa_dev, Xt_dev, y_dev):
        """Grid-search best weights on the dev set."""
        best_acc, best_aw = -1, self.audio_weight
        for aw in np.arange(0.1, 1.0, 0.05):
            tw = 1.0 - aw
            preds = np.argmax(
                aw * self.audio_model.predict_proba(Xa_dev) +
                tw * self.text_model.predict_proba(Xt_dev),
                axis=1,
            )
            acc = accuracy_score(y_dev, preds)
            if acc > best_acc:
                best_acc, best_aw = acc, aw
        self.audio_weight = best_aw
        self.text_weight  = 1.0 - best_aw
        print(f"  Late fusion weights -> audio={best_aw:.2f}, text={1-best_aw:.2f}")
        return best_acc


# ── eval helper ────────────────────────────────────────────────────────────

def _eval(name, model, X_dev, X_test, y_dev, y_test, classes,
          results, models_dir, le, feat_cols, model_type=None):
    dev_acc  = accuracy_score(y_dev,  model.predict(X_dev))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Dev  accuracy: {dev_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, model.predict(X_test), target_names=classes))

    bundle = {
        "model": model, "label_encoder": le,
        "feature_columns": feat_cols,
        "model_type": model_type or name,
    }
    joblib.dump(bundle, models_dir / f"{name}.joblib")
    results.append({"model": name, "dev_accuracy": dev_acc, "test_accuracy": test_acc})


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="store_true",
                        help="Train on 28-class subcategory labels instead of 5-class cluster")
    args = parser.parse_args()

    df = pd.read_csv(FEATURES_CSV)
    label_col = "category" if args.category else "cluster"
    df["label"] = df[label_col]

    audio_cols = [c for c in df.columns if c.startswith("audio_") and c[6:].isdigit()]
    text_cols  = [c for c in df.columns if c.startswith("text_")  and c[5:].isdigit()]
    print(f"Label column   : {label_col}")
    print(f"Audio features : {len(audio_cols)}")
    print(f"Text features  : {len(text_cols)}")

    le = LabelEncoder()
    le.fit(df["label"].values)
    classes = list(le.classes_)
    print(f"Classes ({len(classes)}): {classes}\n")

    Xa_tr, Xt_tr, Xb_tr, y_tr = get_split(df, "train", audio_cols, text_cols, le)
    Xa_dv, Xt_dv, Xb_dv, y_dv = get_split(df, "dev",   audio_cols, text_cols, le)
    Xa_te, Xt_te, Xb_te, y_te = get_split(df, "test",  audio_cols, text_cols, le)
    print(f"Train: {len(y_tr)}  Dev: {len(y_dv)}  Test: {len(y_te)}\n")

    # Clamp k to available features
    k_audio = min(TOP_K_AUDIO, len(audio_cols))
    k_text  = min(TOP_K_TEXT,  len(text_cols))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results = []

    # ── 1. Audio-only ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Training: audio_only (SVM)")
    audio_model = build_audio_model()
    audio_model.named_steps["select"].k = k_audio
    audio_model.fit(Xa_tr, y_tr)
    _eval("audio_only", audio_model, Xa_dv, Xa_te, y_dv, y_te, classes,
          results, MODELS_DIR, le, audio_cols, "audio_only")

    # ── 2. Text-only ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: text_only (Logistic Regression)")
    text_model = build_text_model()
    text_model.named_steps["select"].k = k_text
    text_model.fit(Xt_tr, y_tr)
    _eval("text_only", text_model, Xt_dv, Xt_te, y_dv, y_te, classes,
          results, MODELS_DIR, le, text_cols, "text_only")

    # ── 3. Early fusion ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: early_fusion (MLP on audio + text)")
    early_model = build_early_fusion_model()
    early_model.named_steps["select"].k = k_audio + k_text
    early_model.fit(Xb_tr, y_tr)
    _eval("early_fusion", early_model, Xb_dv, Xb_te, y_dv, y_te, classes,
          results, MODELS_DIR, le, audio_cols + text_cols, "early_fusion")

    # ── 4. Late fusion ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: late_fusion (weighted probability blend)")
    late_model = LateFusionClassifier(audio_model, text_model)
    dev_acc = late_model.tune_weights(Xa_dv, Xt_dv, y_dv)
    late_preds_te = late_model.predict(Xa_te, Xt_te)
    test_acc = accuracy_score(y_te, late_preds_te)
    print(f"  Dev  accuracy: {dev_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(classification_report(y_te, late_preds_te, target_names=classes))
    joblib.dump({
        "model": late_model, "label_encoder": le,
        "audio_columns": audio_cols, "text_columns": text_cols,
        "model_type": "late_fusion",
    }, MODELS_DIR / "late_fusion.joblib")
    results.append({"model": "late_fusion", "dev_accuracy": dev_acc, "test_accuracy": test_acc})

    # ── Summary ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("dev_accuracy", ascending=False)
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 60)
    print("Model comparison (sorted by dev accuracy):")
    print(results_df.to_string(index=False))

    best_row  = results_df.iloc[0]
    best_name = best_row["model"]
    print(f"\nBest model: {best_name}  "
          f"(dev={best_row['dev_accuracy']:.4f}  test={best_row['test_accuracy']:.4f})")

    import shutil
    shutil.copy(MODELS_DIR / f"{best_name}.joblib", MODELS_DIR / "best_model.joblib")
    print(f"Saved best model -> models/best_model.joblib")


if __name__ == "__main__":
    main()
