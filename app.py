"""
Streamlit web app for MIREX Mood Classification.

Run:
    pip install streamlit
    streamlit run app.py
"""

import io
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Mood Classifier",
    page_icon="🎵",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

/* Dark card style */
.mood-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    color: white;
}

.cluster-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.confidence-bar-bg {
    background: #0f3460;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}

.top-result {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e94560;
    margin: 0;
    line-height: 1.1;
}

.sub-label {
    color: #8892b0;
    font-size: 0.85rem;
    margin-top: 4px;
}

hr.fancy {
    border: none;
    border-top: 1px solid #0f3460;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
MODELS_DIR      = Path("models")
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
FEATURES_CSV    = Path("data/features.csv")

CLUSTER_COLORS = {
    "Cluster 1": "#e94560",
    "Cluster 2": "#0f3460",
    "Cluster 3": "#533483",
    "Cluster 4": "#e94560",
    "Cluster 5": "#2ec4b6",
}

CLUSTER_INFO = {
    "Cluster 1": {"emoji": "🔥", "vibe": "Energetic & Aggressive",
                  "desc": "Raw energy, rowdy, fiery and visceral music."},
    "Cluster 2": {"emoji": "🌧️", "vibe": "Dark & Melancholic",
                  "desc": "Brooding, wistful, autumnal and emotionally heavy."},
    "Cluster 3": {"emoji": "☀️", "vibe": "Positive & Upbeat",
                  "desc": "Cheerful, rousing, fun and light-hearted."},
    "Cluster 4": {"emoji": "🎭", "vibe": "Sophisticated & Witty",
                  "desc": "Literate, wry, campy and intellectually playful."},
    "Cluster 5": {"emoji": "💫", "vibe": "Passionate & Intense",
                  "desc": "Deep emotional drive, volatile and expressive."},
}

MODEL_NAMES = {
    "best_model":   "Best Model (auto)",
    "early_fusion": "Early Fusion (Audio + Lyrics)",
    "audio_only":   "Audio Only",
    "text_only":    "Lyrics Only",
    "late_fusion":  "Late Fusion",
}

# ── Loaders ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(name="best_model"):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        return None
    return joblib.load(VECTORIZER_PATH)


@st.cache_data
def load_dataset_stats():
    if not FEATURES_CSV.exists():
        return None
    df = pd.read_csv(FEATURES_CSV, usecols=["cluster", "category", "split",
                                             "artist", "title"])
    return df

# ── Feature extraction ─────────────────────────────────────────────────────

def extract_audio(file_bytes: bytes) -> np.ndarray | None:
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
            y, _  = librosa.effects.trim(y, top_db=30)
            if len(y) < sr:
                y = np.pad(y, (0, sr - len(y)))
            n = 512
            while n > len(y) and n > 32:
                n //= 2
            n = max(n, 32)

            def ms(arr):
                return np.concatenate([arr.mean(axis=-1).ravel(),
                                       arr.std(axis=-1).ravel()])

            mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n)
            mfcc_d  = librosa.feature.delta(mfcc)
            mfcc_d2 = librosa.feature.delta(mfcc, order=2)
            chroma  = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n)
            contrast= librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n)
            zcr     = librosa.feature.zero_crossing_rate(y)
            rms     = librosa.feature.rms(y=y, frame_length=n)
            mel     = librosa.power_to_db(
                        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=n),
                        ref=np.max)
            centroid= librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n)
            bw      = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        return np.nan_to_num(np.concatenate([
            ms(mfcc), ms(mfcc_d), ms(mfcc_d2),
            ms(chroma), ms(contrast), ms(zcr), ms(rms),
            ms(mel), ms(centroid), ms(bw), ms(rolloff), ms(tonnetz),
        ])).astype(np.float32)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None


def get_text_vec(text: str, vectorizer) -> np.ndarray:
    from scipy.sparse import issparse
    X = vectorizer.transform([text])
    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32)


# ── Prediction ─────────────────────────────────────────────────────────────

def run_prediction(bundle, vectorizer, Xa, Xt, top_n=5):
    model      = bundle["model"]
    le         = bundle["label_encoder"]
    model_type = bundle.get("model_type", "early_fusion")
    feat_cols  = bundle.get("feature_columns",
                  bundle.get("audio_columns", []) + bundle.get("text_columns", []))
    n_audio = sum(1 for c in feat_cols if c.startswith("audio_") and c[6:].isdigit())
    n_text  = sum(1 for c in feat_cols if c.startswith("text_")  and c[5:].isdigit())

    try:
        if model_type == "late_fusion":
            if Xa is None or Xt is None:
                return None, "Late fusion needs both audio and lyrics."
            probs = model.predict_proba(Xa.reshape(1,-1), Xt.reshape(1,-1))[0]
        elif model_type == "audio_only":
            if Xa is None:
                return None, "Audio-only model needs an audio file."
            probs = model.predict_proba(Xa.reshape(1,-1))[0]
        elif model_type == "text_only":
            if Xt is None:
                return None, "Lyrics-only model needs lyrics text."
            probs = model.predict_proba(Xt.reshape(1,-1))[0]
        else:  # early_fusion
            ap = Xa.ravel() if Xa is not None else np.zeros(n_audio, np.float32)
            tp = Xt.ravel() if Xt is not None else np.zeros(n_text,  np.float32)
            if Xa is None and Xt is None:
                return None, "Provide audio or lyrics."
            X = np.hstack([ap, tp]).reshape(1,-1)
            probs = model.predict_proba(X)[0]
    except Exception as e:
        return None, str(e)

    top_idx = np.argsort(probs)[::-1][:top_n]
    results = [(le.classes_[i], float(probs[i])) for i in top_idx]
    return results, None


# ── Chart ──────────────────────────────────────────────────────────────────

def draw_bar_chart(results):
    labels = [r[0] for r in results]
    scores = [r[1] * 100 for r in results]
    colors = [CLUSTER_COLORS.get(l, "#e94560") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1],
                   height=0.5, edgecolor="none")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}%", va="center", color="white", fontsize=9)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence %", color="#8892b0", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color("#8892b0")
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("# 🎵 Music Mood Classifier")
    st.markdown("*Upload a song and optionally paste its lyrics to predict the mood cluster.*")
    st.markdown("---")

    # Check models exist
    available_models = [n for n in MODEL_NAMES if (MODELS_DIR / f"{n}.joblib").exists()]
    if not available_models:
        st.error("No trained models found. Run `python train_models.py` first.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_choice = st.selectbox(
            "Model",
            options=available_models,
            format_func=lambda x: MODEL_NAMES.get(x, x),
        )
        top_n = st.slider("Show top N predictions", 1, 5, 3)
        st.markdown("---")
        st.markdown("### 📊 Cluster Guide")
        for cid, info in CLUSTER_INFO.items():
            st.markdown(f"**{info['emoji']} {cid}** — {info['vibe']}")

        # Dataset stats
        df_stats = load_dataset_stats()
        if df_stats is not None:
            st.markdown("---")
            st.markdown("### 📈 Dataset")
            st.markdown(f"**{len(df_stats)}** songs total")
            for sp in ["train", "dev", "test"]:
                n = (df_stats["split"] == sp).sum()
                st.markdown(f"- {sp.capitalize()}: {n}")

    # Load model
    bundle     = load_model(model_choice)
    vectorizer = load_vectorizer()

    if bundle is None:
        st.error(f"Could not load model: {model_choice}")
        st.stop()

    model_type = bundle.get("model_type", "early_fusion")

    # Input section
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🎧 Audio File")
        audio_file = st.file_uploader(
            "Upload MP3, WAV, or FLAC",
            type=["mp3", "wav", "flac", "ogg"],
            label_visibility="collapsed",
        )
        if audio_file:
            st.audio(audio_file)

    with col2:
        st.markdown("### 📝 Lyrics *(optional)*")
        lyrics_text = st.text_area(
            "Paste song lyrics here",
            height=180,
            placeholder="Paste lyrics here to boost accuracy...",
            label_visibility="collapsed",
        )

    st.markdown("")
    predict_btn = st.button("🔍 Classify Mood", use_container_width=True, type="primary")

    # Prediction
    if predict_btn:
        if audio_file is None and not lyrics_text.strip():
            st.warning("Please upload an audio file or enter lyrics.")
            st.stop()

        with st.spinner("Analysing..."):
            Xa, Xt = None, None

            if audio_file is not None:
                file_bytes = audio_file.read()
                Xa = extract_audio(file_bytes)

            if lyrics_text.strip() and vectorizer is not None:
                Xt = get_text_vec(lyrics_text.strip(), vectorizer)

            results, err = run_prediction(bundle, vectorizer, Xa, Xt, top_n)

        if err:
            st.error(f"Prediction failed: {err}")
            st.stop()

        # ── Results ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎭 Results")

        top_cluster, top_conf = results[0]
        info = CLUSTER_INFO.get(top_cluster, {})

        st.markdown(f"""
        <div class="mood-card">
            <div class="sub-label">TOP PREDICTION</div>
            <p class="top-result">{info.get('emoji','')} {top_cluster}</p>
            <div style="color:#c9d1d9; font-size:1.05rem; margin:6px 0 2px;">
                {info.get('vibe', '')}
            </div>
            <div class="sub-label">{info.get('desc','')}</div>
            <div style="margin-top:14px;">
                <div style="color:#8892b0; font-size:0.8rem;">CONFIDENCE</div>
                <div style="font-size:1.6rem; color:#e94560; font-weight:600;">
                    {top_conf*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(results) > 1:
            st.markdown("### All Predictions")
            fig = draw_bar_chart(results)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            for rank, (cluster, conf) in enumerate(results, 1):
                inf = CLUSTER_INFO.get(cluster, {})
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{rank}. {inf.get('emoji','')} {cluster}** — {inf.get('vibe','')}")
                with col_b:
                    st.markdown(f"`{conf*100:.1f}%`")

        # Input summary
        st.markdown("---")
        used = []
        if Xa is not None: used.append("🎧 Audio")
        if Xt is not None: used.append("📝 Lyrics")
        st.caption(f"Model: `{MODEL_NAMES.get(model_choice, model_choice)}` · "
                   f"Input: {' + '.join(used)}")

    # ── Explorer tab ───────────────────────────────────────────────────────
    with st.expander("📂 Browse Dataset Songs"):
        df_stats = load_dataset_stats()
        if df_stats is not None:
            cluster_filter = st.multiselect(
                "Filter by cluster",
                options=sorted(df_stats["cluster"].unique()),
                default=sorted(df_stats["cluster"].unique()),
            )
            filtered = df_stats[df_stats["cluster"].isin(cluster_filter)]
            st.dataframe(
                filtered[["title", "artist", "category", "cluster", "split"]].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )
        else:
            st.info("Run the full pipeline first to see dataset stats.")


if __name__ == "__main__":
    main()