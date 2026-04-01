# ============================================================
#  model_utils.py
#  Fungsi inti yang dipakai bersama oleh app.py dan
#  streamlit_app.py.
#
#  Alur training yang benar:
#    Step 1 — Split dataset → training set & testing set (80:20)
#    Step 2 — Training awal pada training set
#    Step 3 — Cross-Validation 5-fold HANYA pada training set
#    Step 4 — Final test pada testing set (data yang belum
#              pernah dilihat model sama sekali)
# ============================================================

import os
import re
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, clone
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
#  TRAINING
#  Alur: Train → Cross-Validation → Final Test
# ============================================================

def train_model(csv_path: str, progress_callback=None) -> dict:
    """
    Melatih model dengan alur yang benar:
      Step 1 : Split dataset (80% train / 20% test)
      Step 2 : Training model pada training set
      Step 3 : 5-Fold Cross-Validation pada training set saja
      Step 4 : Evaluasi final pada testing set

    Parameter
    ---------
    csv_path          : path ke file dataset CSV
    progress_callback : fungsi(str) untuk pesan progres (opsional)

    Return
    ------
    dict metrik: accuracy, precision, recall, f1,
                 cv_scores, cv_mean, cv_std,
                 train_size, test_size
    """

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # ── Step 0: Muat dataset ─────────────────────────────────
    log("Memuat dataset...")
    df = pd.read_csv(csv_path)

    if 'headline' not in df.columns or 'clickbait' not in df.columns:
        raise ValueError(
            "Kolom 'headline' atau 'clickbait' tidak ditemukan dalam CSV."
        )

    # ── Preprocessing ────────────────────────────────────────
    log("Melakukan preprocessing teks...")
    df['text_clean'] = df['headline'].apply(preprocess)

    X = df['text_clean']
    y = df['clickbait']

    # ── Step 1: Split dataset ────────────────────────────────
    log(
        "Step 1/4 — Membagi dataset...\n"
        "  Training set : 80%\n"
        "  Testing set  : 20%\n"
        "  (Testing set dikunci, tidak disentuh hingga Step 4)"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ── Definisi pipeline ────────────────────────────────────
    # Dibuat sekali, di-clone untuk CV agar tidak bocor
    base_pipeline = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=5),
        GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    )

    # ── Step 2: Training pada training set ───────────────────
    log(
        f"Step 2/4 — Training model...\n"
        f"  Data training : {len(X_train)} sampel\n"
        f"  Proses ini membutuhkan beberapa menit, harap tunggu."
    )
    # Clone pipeline baru untuk training final (bebas dari CV)
    final_pipeline = clone(base_pipeline)
    final_pipeline.fit(X_train, y_train)

    # ── Step 3: Cross-Validation pada training set ───────────
    log(
        "Step 3/4 — 5-Fold Cross-Validation pada training set...\n"
        "  CV dilakukan HANYA pada training set (X_train, y_train)\n"
        "  Testing set tetap dikunci dan belum disentuh.\n"
        "  Proses ini membutuhkan waktu tambahan, harap tunggu."
    )
    # Clone pipeline baru untuk CV agar tidak memakai model
    # yang sudah di-fit pada Step 2
    cv_pipeline = clone(base_pipeline)
    cv_scores = cross_val_score(
        cv_pipeline,
        X_train,    # ← hanya training set
        y_train,    # ← hanya training set
        cv=5,
        scoring='accuracy'
    )

    cv_scores_pct = [round(s * 100, 2) for s in cv_scores.tolist()]
    cv_mean       = round(cv_scores.mean() * 100, 2)
    cv_std        = round(cv_scores.std()  * 100, 2)

    log(
        f"  CV Scores : {cv_scores_pct}\n"
        f"  Mean      : {cv_mean}%\n"
        f"  Std Dev   : ± {cv_std}%"
    )

    # ── Step 4: Final test pada testing set ──────────────────
    log(
        "Step 4/4 — Evaluasi final pada testing set...\n"
        f"  Data testing : {len(X_test)} sampel\n"
        "  Testing set baru digunakan di tahap ini."
    )
    y_pred = final_pipeline.predict(X_test)

    accuracy  = round(accuracy_score(y_test, y_pred)  * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall    = round(recall_score(y_test, y_pred)    * 100, 2)
    f1        = round(f1_score(y_test, y_pred)        * 100, 2)

    metrics = {
        # Metrik Final Test (Step 4)
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        # Metrik Cross-Validation (Step 3)
        "cv_scores":  cv_scores_pct,
        "cv_mean":    cv_mean,
        "cv_std":     cv_std,
        # Info ukuran data
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "total_size": len(X),
    }

    # ── Simpan model final dan metrik ────────────────────────
    log("Menyimpan model ke disk...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_pipeline, f)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    log("✅ Semua tahapan selesai. Model berhasil disimpan.")
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
