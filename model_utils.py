# ============================================================
#  model_utils.py
#  Fungsi inti yang dipakai bersama oleh app.py dan
#  streamlit_app.py:
#    - Preprocessing teks
#    - Pelatihan model  (hanya dipakai app.py / VSCode)
#    - Simpan & muat model
#    - Prediksi
# ============================================================

import os
import re
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ── Path file model ─────────────────────────────────────────
MODEL_PATH   = "saved_model.pkl"
METRICS_PATH = "saved_metrics.pkl"


# ============================================================
#  PREPROCESSING
#  Sama persis dengan klasifikasi_remake.py
# ============================================================

def lowercase_text(sentence: str) -> str:
    return sentence.lower()


def normalize_text(sentence: str) -> str:
    return re.sub(r'\s+', ' ', sentence).strip()


def preprocess(text: str) -> str:
    """Tahap 1: lowercase → Tahap 2: normalisasi spasi."""
    text = lowercase_text(text)
    text = normalize_text(text)
    return text


# ============================================================
#  TRAINING  (hanya dipanggil dari app.py / VSCode)
# ============================================================

def train_model(csv_path: str, progress_callback=None) -> dict:
    """
    Melatih model dari file CSV dan menyimpannya ke disk.

    Parameter
    ---------
    csv_path          : path ke file dataset CSV
    progress_callback : fungsi(str) untuk pesan progres (opsional)

    Return
    ------
    dict metrik evaluasi: accuracy, precision, recall, f1,
    train_size, test_size
    """

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # 1. Muat dataset
    log("Memuat dataset...")
    df = pd.read_csv(csv_path)

    if 'headline' not in df.columns or 'clickbait' not in df.columns:
        raise ValueError(
            "Kolom 'headline' atau 'clickbait' tidak ditemukan dalam CSV."
        )

    # 2. Preprocessing
    log("Melakukan preprocessing teks...")
    df['text_clean'] = df['headline'].apply(preprocess)

    X = df['text_clean']
    y = df['clickbait']

    # 3. Split 80:20 dengan stratifikasi
    log("Membagi dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Bangun pipeline
    log(
        f"Melatih model dengan {len(X_train)} data training...\n"
        "Proses ini membutuhkan beberapa menit, harap tunggu."
    )
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=5),
        GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    )
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi
    log("Mengevaluasi model pada testing set...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":   round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision":  round(precision_score(y_test, y_pred) * 100, 2),
        "recall":     round(recall_score(y_test, y_pred)    * 100, 2),
        "f1":         round(f1_score(y_test, y_pred)        * 100, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    # 6. Simpan ke disk
    log("Menyimpan model ke disk...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    log("✅ Model berhasil disimpan.")
    return metrics


# ============================================================
#  MUAT / HAPUS MODEL
# ============================================================

def load_model():
    """
    Muat model dan metrik dari disk.
    Return (model, metrics) atau (None, None) jika belum ada.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(METRICS_PATH, 'rb') as f:
            metrics = pickle.load(f)
        return model, metrics
    return None, None


def delete_model():
    """Hapus file model dan metrik dari disk."""
    for path in [MODEL_PATH, METRICS_PATH]:
        if os.path.exists(path):
            os.remove(path)


def model_exists() -> bool:
    """Cek apakah file model sudah tersedia."""
    return os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)


# ============================================================
#  PREDIKSI
# ============================================================

def predict(model, text: str) -> dict:
    """
    Prediksi satu judul berita.

    Return dict:
      label      : 1 (clickbait) atau 0 (non-clickbait)
      confidence : persentase keyakinan model
      text_clean : teks setelah preprocessing
    """
    clean      = preprocess(text)
    label      = model.predict([clean])[0]
    proba      = model.predict_proba([clean])[0]
    confidence = round(max(proba) * 100, 1)
    return {
        "label":      int(label),
        "confidence": confidence,
        "text_clean": clean,
    }
