"""
Evaluate the best saved model on the MIREX Mood test split.

Generates:
  reports/classification_report.txt
  reports/confusion_matrix.png        (normalised heatmap)
  reports/roc_curves.png              (one-vs-rest per class)
  reports/per_artist_accuracy.csv     (spot artist bias)
  reports/modality_comparison.png     (audio vs text vs fusion bar chart)
  reports/category_breakdown.csv      (accuracy per fine-grained subcategory)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from train_models import LateFusionClassifier

FEATURES_CSV = Path("data/features.csv")
MODELS_DIR   = Path("models")
REPORTS_DIR  = Path("reports")


# ── loaders ────────────────────────────────────────────────────────────────

def load_split(df, split, audio_cols, text_cols, le):
    mask = df["split"] == split
    Xa = df.loc[mask, audio_cols].values.astype(float)
    Xt = df.loc[mask, text_cols].values.astype(float)
    Xb = np.hstack([Xa, Xt])
    y  = le.transform(df.loc[mask, "label"].values)
    return Xa, Xt, Xb, y, df[mask].reset_index(drop=True)


# ── plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".1f"], ["Counts", "Row-normalised (%)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual",    fontsize=12)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(y_true_bin, y_prob, classes, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_modality_comparison(model_accs: dict, path):
    names  = list(model_accs.keys())
    accs   = [model_accs[n] for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, min(1.0, max(accs) + 0.1))
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Modality Comparison — Test Set", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── per-artist and per-category analysis ───────────────────────────────────

def per_group_accuracy(df_test, pred_labels, group_col):
    rows = []
    for grp in df_test[group_col].unique():
        mask = df_test[group_col] == grp
        true = df_test.loc[mask, "label"].values
        pred = pred_labels[mask]
        acc  = (true == pred).mean()
        rows.append({group_col: grp, "n_samples": int(mask.sum()),
                     "accuracy": round(acc, 4)})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


# ── prediction dispatch ────────────────────────────────────────────────────

def _predict_bundle(bundle, Xa, Xt, Xb):
    model      = bundle["model"]
    model_type = bundle.get("model_type", "early_fusion")
    if model_type == "late_fusion":
        return model.predict(Xa, Xt), model.predict_proba(Xa, Xt)
    elif model_type == "audio_only":
        return model.predict(Xa), model.predict_proba(Xa)
    elif model_type == "text_only":
        return model.predict(Xt), model.predict_proba(Xt)
    else:
        return model.predict(Xb), model.predict_proba(Xb)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(FEATURES_CSV)

    # Determine label column (cluster or category)
    best_bundle = joblib.load(MODELS_DIR / "best_model.joblib")
    le = best_bundle["label_encoder"]
    classes = list(le.classes_)

    # Identify which label column matches the encoder
    if classes[0].startswith("Cluster"):
        label_col = "cluster"
    else:
        label_col = "category"
    df["label"] = df[label_col]

    audio_cols = [c for c in df.columns if c.startswith("audio_") and c[6:].isdigit()]
    text_cols  = [c for c in df.columns if c.startswith("text_")  and c[5:].isdigit()]

    Xa_te, Xt_te, Xb_te, y_te, df_te = load_split(df, "test", audio_cols, text_cols, le)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    preds, probs = _predict_bundle(best_bundle, Xa_te, Xt_te, Xb_te)

    # ── Classification report ──────────────────────────────────────────────
    report = classification_report(y_te, preds, target_names=classes, digits=4)
    print("\nClassification Report (best model):\n")
    print(report)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_te, preds)
    plot_confusion_matrix(cm, classes, REPORTS_DIR / "confusion_matrix.png")

    # ── ROC curves ─────────────────────────────────────────────────────────
    if probs is not None and len(classes) > 2:
        y_bin = label_binarize(y_te, classes=list(range(len(classes))))
        plot_roc_curves(y_bin, probs, classes, REPORTS_DIR / "roc_curves.png")
        macro_auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        print(f"\nMacro-average AUC (OvR): {macro_auc:.4f}")

    # ── Per-artist accuracy ────────────────────────────────────────────────
    if "artist" in df_te.columns:
        art_df = per_group_accuracy(df_te, le.inverse_transform(preds), "artist")
        print("\nPer-artist accuracy (top 10):")
        print(art_df.head(10).to_string(index=False))
        art_df.to_csv(REPORTS_DIR / "per_artist_accuracy.csv", index=False)

    # ── Category breakdown (if model trained on cluster) ───────────────────
    if label_col == "cluster" and "category" in df_te.columns:
        pred_labels = le.inverse_transform(preds)
        cat_rows = []
        for cat in df_te["category"].unique():
            mask = df_te["category"] == cat
            true = df_te.loc[mask, "label"].values
            pred = pred_labels[mask]
            acc  = (true == pred).mean()
            cat_rows.append({"category": cat, "cluster": true[0],
                              "n": int(mask.sum()), "accuracy": round(acc, 4)})
        cat_df = pd.DataFrame(cat_rows).sort_values("accuracy", ascending=False)
        print("\nAccuracy by subcategory (top 10):")
        print(cat_df.head(10).to_string(index=False))
        cat_df.to_csv(REPORTS_DIR / "category_breakdown.csv", index=False)

    # ── Modality comparison ────────────────────────────────────────────────
    model_accs = {}
    for name in ["audio_only", "text_only", "early_fusion", "late_fusion"]:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        b = joblib.load(path)
        p, _ = _predict_bundle(b, Xa_te, Xt_te, Xb_te)
        model_accs[name] = accuracy_score(y_te, p)

    if model_accs:
        plot_modality_comparison(model_accs, REPORTS_DIR / "modality_comparison.png")

    print(f"\nAll reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
