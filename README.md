# 🎵 Multimodal Mood Recognition — MIREX 2013

Classify the **mood** of songs using the MIREX 2013 Mood Classification dataset,
combining **audio** (acoustic features) and **text** (lyrics) signals.

## Mood Clusters (5 classes)
`Cluster 1 · Cluster 2 · Cluster 3 · Cluster 4 · Cluster 5`

## Subcategories (28 classes)
Boisterous · Passionate · Literate · Bittersweet · Campy · Confident · Wry ·
Autumnal · Brooding · Amiable-good natured · Sweet · Wistful · Whimsical ·
Rousing · Rollicking · Cheerful · Poignant · Witty · Visceral · Tense-Anxious ·
Rowdy · Fun · Silly · Volatile · Fiery · Intense · Humorous · Agressive

---

## Dataset Structure
Dataset link https://www.kaggle.com/datasets/imsparsh/multimodal-mirex-emotion-dataset/discussion?sort=hotness

```
dataset/
├── Audio/          # 903 MP3 clips (30 sec each, 001.mp3 – 903.mp3)
├── Lyrics/         # 764 lyrics files (001.txt – 903.txt)
├── MIDIs/          # 196 MIDI files  (001.mid – 903.mid)
├── categories.txt  # 903 subcategory labels (one per line)
├── clusters.txt    # 903 cluster labels (one per line)
└── dataset info.csv
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python build_dataset.py        # build data/dataset.csv with train/dev/test splits
python extract_features.py     # audio + lyrics features -> data/features.csv
python train_models.py         # train 4 models -> models/
python evaluate.py             # reports + plots -> reports/
```

To train on 28-class subcategory labels instead of 5 clusters:
```bash
python train_models.py --category
```

### 3. Predict on new songs
```bash
# Audio + lyrics (multimodal — best accuracy)
python predict.py dataset/Audio/001.mp3 --text "$(cat dataset/Lyrics/001.txt)"

# Audio only
python predict.py dataset/Audio/001.mp3

# Lyrics only
python predict.py --text "I can't help falling in love with you"

# Top-3 predictions
python predict.py dataset/Audio/001.mp3 --top 3

# Batch over all test audio
python predict.py dataset/Audio/ --batch
```

---

## Project Structure

```
AML_PROJECT/
├── build_dataset.py       Parse labels, locate files, create stratified splits
├── extract_features.py    Audio (librosa) + lyrics (TF-IDF) feature extraction
├── train_models.py        Train 4 unimodal + multimodal classifiers
├── evaluate.py            Confusion matrix, ROC, per-artist accuracy
├── predict.py             CLI inference on new songs
├── multimodal_emotion.ipynb   Interactive walkthrough + EDA
├── dataset/               Raw data (Audio, Lyrics, MIDIs, labels)
├── data/                  Built by pipeline (dataset.csv, features.csv)
├── models/                Trained models (.joblib)
└── reports/               Evaluation outputs (plots, CSVs, text reports)
```

---

## Models

| Model | Modality | Algorithm |
|-------|----------|-----------|
| `audio_only` | Audio only | SVM (RBF kernel) + SelectKBest |
| `text_only` | Lyrics only | Logistic Regression + TF-IDF |
| `early_fusion` | Audio + Lyrics | MLP on concatenated features |
| `late_fusion` | Audio + Lyrics | Weighted probability blend (tuned on dev) |

---

## Key Design Decisions

- **Stratified train/dev/test split** (70/15/15) on cluster labels — ensures
  each cluster is proportionally represented in every subset.
- **TF-IDF fit on train only** — no data leakage into dev/test for lyrics.
- **Class weights = "balanced"** — handles any class imbalance across clusters.
- **Late fusion weights tuned on dev set** — grid search over audio/text weight ratio.
- **Resume support** in `extract_features.py` — safe to interrupt and restart.
- **Sample rate 22 050 Hz** — matches the MIREX standard for music analysis.
