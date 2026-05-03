"""
Build the MIREX Mood dataset CSV.

Dataset layout (your "dataset" folder):
  dataset/
    Audio/          <- 001.mp3 ... 903.mp3
    Lyrics/         <- 001.txt ... 903.txt  (764 available)
    MIDIs/          <- 001.mid ... 903.mid  (196 available)
    categories.txt  <- one subcategory label per song (903 lines)
    clusters.txt    <- one cluster label per song (903 lines)
    dataset info.csv

Output: data/dataset.csv

Run:
    python build_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("dataset")
AUDIO_DIR   = DATA_DIR / "Audio"
LYRICS_DIR  = DATA_DIR / "Lyrics"
MIDI_DIR    = DATA_DIR / "MIDIs"
CATS_FILE   = DATA_DIR / "categories.txt"
CLUST_FILE  = DATA_DIR / "clusters.txt"
INFO_CSV    = DATA_DIR / "dataset info.csv"
OUT_FILE    = Path("data/dataset.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.15   # 15% for test
DEV_SIZE     = 0.15   # 15% for dev (from remaining)


def load_labels() -> tuple[list[str], list[str]]:
    with open(CATS_FILE, encoding="utf-8", errors="replace") as f:
        categories = [l.strip() for l in f if l.strip()]
    with open(CLUST_FILE, encoding="utf-8", errors="replace") as f:
        clusters = [l.strip() for l in f if l.strip()]
    return categories, clusters


def load_metadata() -> dict[str, dict]:
    meta = {}
    if not INFO_CSV.exists():
        return meta
    try:
        df = pd.read_csv(INFO_CSV, encoding="utf-16", sep=";", on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            fname = str(row.get("Filename", "")).strip()
            if fname:
                stem = Path(fname).stem.zfill(3)
                meta[stem] = {
                    "title":  str(row.get("Title",  "")).strip(),
                    "artist": str(row.get("Artist", "")).strip(),
                    "album":  str(row.get("Album",  "")).strip(),
                    "year":   str(row.get("Year",   "")).strip(),
                }
    except Exception as e:
        print(f"  Warning: could not read metadata CSV ({e})")
    return meta


def assign_splits(n: int, labels: list[str]) -> list[str]:
    indices = np.arange(n)
    idx_tr_dv, idx_te = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    lbl_tr_dv = [labels[i] for i in idx_tr_dv]
    idx_tr, idx_dv = train_test_split(
        idx_tr_dv,
        test_size=DEV_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=lbl_tr_dv,
    )
    splits = ["train"] * n
    for i in idx_dv:
        splits[i] = "dev"
    for i in idx_te:
        splits[i] = "test"
    return splits


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Folder '{DATA_DIR}' not found.")
        return

    print("Loading labels ...")
    categories, clusters = load_labels()
    n = len(categories)
    print(f"  {n} songs")
    print(f"  Clusters  : {dict(Counter(clusters))}")
    print(f"  Categories: {len(set(categories))} unique")

    print("\nLoading metadata ...")
    meta = load_metadata()
    print(f"  Metadata entries: {len(meta)}")

    print("\nBuilding rows ...")
    rows = []
    missing_audio = 0

    for idx in range(n):
        file_id = str(idx + 1).zfill(3)
        audio_path  = AUDIO_DIR  / f"{file_id}.mp3"
        lyrics_path = LYRICS_DIR / f"{file_id}.txt"
        midi_path   = MIDI_DIR   / f"{file_id}.mid"

        if not audio_path.exists():
            missing_audio += 1
            continue

        info = meta.get(file_id, {})
        rows.append({
            "file_id":     file_id,
            "audio_path":  str(audio_path),
            "lyrics_path": str(lyrics_path) if lyrics_path.exists() else "",
            "midi_path":   str(midi_path)   if midi_path.exists()   else "",
            "category":    categories[idx],
            "cluster":     clusters[idx],
            "title":       info.get("title",  ""),
            "artist":      info.get("artist", ""),
            "album":       info.get("album",  ""),
            "year":        info.get("year",   ""),
        })

    if missing_audio:
        print(f"  Warning: {missing_audio} audio file(s) not found and skipped.")

    print(f"  Usable songs: {len(rows)}")

    label_list = [r["cluster"] for r in rows]
    splits = assign_splits(len(rows), label_list)
    for row, sp in zip(rows, splits):
        row["split"] = sp

    df = pd.DataFrame(rows)
    sc = df["split"].value_counts()
    print(f"\n  Splits -> train: {sc.get('train',0)}  dev: {sc.get('dev',0)}  test: {sc.get('test',0)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(df)} rows -> {OUT_FILE}")
    print("\nCluster distribution:")
    print(df["cluster"].value_counts().to_string())
    print("\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nSongs with lyrics : {(df['lyrics_path'] != '').sum()}")
    print(f"Songs with MIDI   : {(df['midi_path']   != '').sum()}")


if __name__ == "__main__":
    main()
