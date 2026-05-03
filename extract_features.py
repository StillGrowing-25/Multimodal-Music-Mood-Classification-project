"""
Feature extraction for MIREX Mood Classification - audio + lyrics (multimodal).

Audio features (~428 values per clip):
  MFCCs (40) + delta + delta-delta  -> mean & std  (240)
  Chroma STFT (12)                  -> mean & std   (24)
  Spectral contrast (7)             -> mean & std   (14)
  ZCR + RMS                         -> mean & std    (4)
  Mel spectrogram (64)              -> mean & std  (128)
  Spectral centroid/bw/rolloff      -> mean & std    (6)
  Tonnetz (6)                       -> mean & std   (12)
                                               Total: 428

Text features (lyrics):
  TF-IDF (max_features=5000), fit ONLY on train split to avoid leakage.
  Songs without lyrics get a zero vector.

Output:
  data/features.csv        -- one row per song, columns:
      file_id, audio_path, lyrics_path, category, cluster, split,
      audio_0 ... audio_N,
      text_0  ... text_M

Run:
  python extract_features.py
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

DATASET_CSV     = Path("data/dataset.csv")
FEATURES_CSV    = Path("data/features.csv")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")

SR          = 22_050   # MIREX standard is 22 kHz
N_MFCC      = 40
N_MELS      = 64
TOP_DB      = 30
MIN_LEN     = SR       # pad to at least 1 second
DEFAULT_FFT = 512
FLUSH_EVERY = 100
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# ── audio helpers ──────────────────────────────────────────────────────────

def _safe_fft(n: int, desired: int = DEFAULT_FFT) -> int:
    while desired > n and desired > 32:
        desired //= 2
    return max(desired, 32)


def _ms(arr: np.ndarray) -> np.ndarray:
    """Concatenate mean and std across time axis."""
    return np.concatenate([arr.mean(axis=-1).ravel(), arr.std(axis=-1).ravel()])


def extract_audio_features(file_path: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        y, _  = librosa.effects.trim(y, top_db=TOP_DB)
        if len(y) < MIN_LEN:
            y = np.pad(y, (0, MIN_LEN - len(y)))

        n = _safe_fft(len(y))

        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n)
        mfcc_d  = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n)
        zcr      = librosa.feature.zero_crossing_rate(y)
        rms      = librosa.feature.rms(y=y, frame_length=n)
        mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=n)
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n)
        bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n)
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n)
        harmonic = librosa.effects.harmonic(y)
        tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)

    return np.nan_to_num(np.concatenate([
        _ms(mfcc), _ms(mfcc_d), _ms(mfcc_d2),
        _ms(chroma), _ms(contrast),
        _ms(zcr), _ms(rms),
        _ms(mel_db),
        _ms(centroid), _ms(bandwidth), _ms(rolloff),
        _ms(tonnetz),
    ]))


# ── worker ─────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[dict | None, str | None]:
    (idx, file_id, audio_path, lyrics_path,
     midi_path, category, cluster, title, artist, split) = args
    try:
        feats = extract_audio_features(audio_path)
        row = {
            "__idx":      idx,
            "file_id":    file_id,
            "audio_path": audio_path,
            "lyrics_path":lyrics_path,
            "midi_path":  midi_path,
            "category":   category,
            "cluster":    cluster,
            "title":      title,
            "artist":     artist,
            "split":      split,
        }
        for i, v in enumerate(feats):
            row[f"audio_{i}"] = v
        return row, None
    except Exception as e:
        return None, f"{audio_path}: {e}"


# ── lyrics reader ──────────────────────────────────────────────────────────

def _read_lyrics(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def extract_text_features(df: pd.DataFrame) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit TF-IDF on train split only, transform all.
    Songs without lyrics (empty string) get a zero row automatically.
    """
    train_mask = df["split"] == "train"
    texts = df["lyrics_path"].apply(_read_lyrics)

    vectorizer = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    vectorizer.fit(texts[train_mask].fillna(""))
    X_text = vectorizer.transform(texts.fillna(""))
    if issparse(X_text):
        X_text = X_text.toarray()
    return X_text.astype(np.float32), vectorizer


# ── flush helper ───────────────────────────────────────────────────────────

def _flush(buffer: list[dict], path: Path, header: bool) -> None:
    pd.DataFrame(buffer).to_csv(path, mode="a", header=header, index=False)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    df_all = pd.read_csv(DATASET_CSV)
    print(f"Dataset: {len(df_all)} songs")

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = True

    already_done: set[str] = set()
    if FEATURES_CSV.exists():
        existing = pd.read_csv(FEATURES_CSV, usecols=["audio_path"])
        already_done = set(existing["audio_path"])
        df_todo = df_all[~df_all["audio_path"].isin(already_done)].reset_index(drop=True)
        write_header = False
        print(f"Resuming: {len(already_done)} done, {len(df_todo)} remaining")
    else:
        df_todo = df_all.copy()
        print(f"Starting fresh: {len(df_todo)} files")

    if not df_todo.empty:
        args_list = [
            (i,
             row["file_id"], row["audio_path"], row.get("lyrics_path",""),
             row.get("midi_path",""), row["category"], row["cluster"],
             row.get("title",""), row.get("artist",""), row["split"])
            for i, row in df_todo.iterrows()
        ]
        buffer: list[dict] = []
        errors: list[str]  = []

        print(f"Workers: {NUM_WORKERS}  |  Flush every {FLUSH_EVERY}\n")

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_worker, a): a for a in args_list}
            with tqdm(total=len(args_list), desc="Audio features", unit="file") as pbar:
                for future in as_completed(futures):
                    row, err = future.result()
                    if err:
                        errors.append(err)
                    else:
                        buffer.append(row)
                    pbar.update(1)
                    if len(buffer) >= FLUSH_EVERY:
                        _flush(buffer, FEATURES_CSV, write_header)
                        write_header = False
                        buffer.clear()
        if buffer:
            _flush(buffer, FEATURES_CSV, write_header)
        if errors:
            print(f"\n{len(errors)} file(s) failed:")
            for e in errors[:10]:
                print(f"  {e}")

    # ── Text features ───────────────────────────────────────────────────────
    print("\nLoading audio features for text extraction ...")
    df_feat = pd.read_csv(FEATURES_CSV).sort_values("file_id").reset_index(drop=True)

    print("Extracting text features (TF-IDF on lyrics) ...")
    X_text, vectorizer = extract_text_features(df_feat)

    n_audio = len([c for c in df_feat.columns if c.startswith("audio_")])
    text_cols = [f"text_{i}" for i in range(X_text.shape[1])]
    df_text = pd.DataFrame(X_text, columns=text_cols)
    df_combined = pd.concat([df_feat.reset_index(drop=True), df_text], axis=1)
    df_combined = df_combined.drop(columns=["__idx"], errors="ignore")
    df_combined.to_csv(FEATURES_CSV, index=False)

    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nFeatures saved -> {FEATURES_CSV}")
    print(f"  Audio features : {n_audio}")
    print(f"  Text features  : {X_text.shape[1]}")
    print(f"  Total columns  : {n_audio + X_text.shape[1]}")
    print(f"  Rows           : {len(df_combined)}")
    print(f"\nTF-IDF vectorizer saved -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
