"""
Predict mood cluster from an audio file and/or lyrics text.

Usage:
  # Audio + lyrics (multimodal -- best accuracy)
  python predict.py path/to/song.mp3 --text "lyrics go here..."

  # Audio only
  python predict.py path/to/song.mp3

  # Text (lyrics) only
  python predict.py --text "lyrics go here..."

  # Show top-3 predictions
  python predict.py song.mp3 --text "..." --top 3

  # Batch mode (folder of audio files; looks for matching .txt lyrics)
  python predict.py dataset/Audio/ --batch

  # Use a specific model (default: best_model)
  python predict.py song.mp3 --model audio_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR      = Path("models")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def load_bundle(name: str = "best_model"):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        sys.exit(f"Model not found at {path}. Run train_models.py first.")
    return joblib.load(path)


def get_audio_features(audio_path: str) -> np.ndarray:
    from extract_features import extract_audio_features
    return extract_audio_features(audio_path)


def get_text_features(text: str, vectorizer) -> np.ndarray:
    from scipy.sparse import issparse
    X = vectorizer.transform([text])
    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


def predict_mood(
    audio_path: str | None,
    text: str | None,
    bundle: dict,
    vectorizer,
    top_n: int = 1,
) -> list[tuple[str, float]]:
    model      = bundle["model"]
    le         = bundle["label_encoder"]
    model_type = bundle.get("model_type", "early_fusion")

    Xa = get_audio_features(audio_path) if audio_path else None
    Xt = get_text_features(text, vectorizer) if (text and vectorizer) else None

    feat_cols = bundle.get("feature_columns", bundle.get("audio_columns", []))
    audio_n   = sum(1 for c in feat_cols if c.startswith("audio_"))

    if model_type == "late_fusion":
        audio_cols = bundle.get("audio_columns", [c for c in feat_cols if c.startswith("audio_")])
        text_cols  = bundle.get("text_columns",  [c for c in feat_cols if c.startswith("text_")])
        if Xa is None or Xt is None:
            sys.exit("Late fusion model requires both audio and --text inputs.")
        probs = model.predict_proba(
            Xa.reshape(1, -1),
            Xt.reshape(1, -1),
        )[0]
    elif model_type == "audio_only":
        if Xa is None:
            sys.exit("audio_only model requires an audio file.")
        probs = model.predict_proba(Xa.reshape(1, -1))[0]
    elif model_type == "text_only":
        if Xt is None:
            sys.exit("text_only model requires --text input.")
        probs = model.predict_proba(Xt.reshape(1, -1))[0]
    else:  # early_fusion
        feat_cols = bundle.get("feature_columns", [])
        n_audio = sum(1 for c in feat_cols if c.startswith("audio_") and c[6:].isdigit())
        n_text  = sum(1 for c in feat_cols if c.startswith("text_")  and c[5:].isdigit())

        audio_part = Xa.ravel() if Xa is not None else np.zeros(n_audio, dtype=np.float32)
        text_part  = Xt.ravel() if Xt is not None else np.zeros(n_text,  dtype=np.float32)

        if Xa is None and Xt is None:
            sys.exit("Provide at least one of: audio file, --text")

        X = np.hstack([audio_part, text_part]).reshape(1, -1)
        probs = model.predict_proba(X)[0]

    top_idx = np.argsort(probs)[::-1][:top_n]
    return [(le.classes_[i], float(probs[i])) for i in top_idx]


def render_bar(conf: float, width: int = 28) -> str:
    filled = round(conf * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {conf * 100:.1f}%"


def print_results(source: str, results: list[tuple]):
    print(f"\n>>  {source}")
    for rank, (mood, conf) in enumerate(results, 1):
        marker = "->" if rank == 1 else "  "
        bar = render_bar(conf)
        print(f"  {marker} #{rank}  {mood:<15}  {bar}")


def batch_predict(folder: Path, bundle: dict, vectorizer, top_n: int):
    exts = {".mp3", ".wav", ".flac", ".ogg"}
    audio_files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in exts)
    if not audio_files:
        sys.exit(f"No audio files found in {folder}")

    records = []
    for f in audio_files:
        txt_file = f.with_suffix(".txt")
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip() \
               if txt_file.exists() else None
        try:
            results = predict_mood(str(f), text, bundle, vectorizer, top_n=1)
            print_results(f.name + (f" | {text[:60]}" if text else ""), results)
            mood, conf = results[0]
            records.append({"file": str(f), "text_available": text is not None,
                            "mood": mood, "confidence": conf})
        except Exception as e:
            print(f"  Skipped {f.name}: {e}")

    out = Path("reports/batch_predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nBatch results saved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="MIREX Mood predictor")
    parser.add_argument("path",   nargs="?", help="Audio file or folder (--batch)")
    parser.add_argument("--text", type=str, default=None, help="Lyrics text")
    parser.add_argument("--batch",action="store_true",   help="Batch mode over a folder")
    parser.add_argument("--top",  type=int, default=1,   help="Show top-N predictions")
    parser.add_argument("--model",type=str, default="best_model",
                        help="Model name in models/ (default: best_model)")
    args = parser.parse_args()

    if args.path is None and args.text is None:
        parser.print_help()
        sys.exit(1)

    bundle     = load_bundle(args.model)
    vectorizer = joblib.load(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else None

    if args.batch:
        batch_predict(Path(args.path), bundle, vectorizer, args.top)
    else:
        results = predict_mood(args.path, args.text, bundle, vectorizer, args.top)
        src = args.path or ""
        if args.text:
            src += f'  "{args.text[:70]}"'
        print_results(src, results)

        top_mood, top_conf = results[0]
        print(f"\nPredicted: {top_mood}  ({top_conf * 100:.2f}%)")


if __name__ == "__main__":
    main()
