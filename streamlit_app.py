# ============================================================
#  streamlit_app.py  —  Aplikasi Demo (Streamlit)
#
#  Model dibaca dari saved_model.pkl (hasil training VSCode).
#  Menampilkan metrik: CV (Step 3) + Final Test (Step 4).
#
#  Jalankan lokal : streamlit run streamlit_app.py
#  Deploy         : https://streamlit.io/cloud
#  Python         : 3.11
# ============================================================

import streamlit as st
from model_utils import load_model, model_exists, predict

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Clickbait",
    page_icon="🔍",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    .main-header {
        background: linear-gradient(135deg, #1A237E, #3949AB);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; letter-spacing: 1px; }
    .main-header p  { margin: 8px 0 0; opacity: 0.85; font-size: 0.9rem; }

    .step-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 16px 0 8px 0;
    }

    /* Metrik final test */
    .metric-box {
        background: #F0F4FF;
        border: 1px solid #C5CAE9;
        border-radius: 12px;
        padding: 14px 8px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1A237E;
        line-height: 1.2;
    }
    .metric-box .lbl {
        font-size: 0.72rem;
        color: #546E7A;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metrik cross-validation */
    .cv-box {
        background: #F3F3FF;
        border: 1px solid #C5CAE9;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }
    .cv-box .cv-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #283593;
    }
    .cv-box .cv-lbl {
        font-size: 0.72rem;
        color: #546E7A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Hasil deteksi */
    .result-clickbait {
        background: #FFF3F3;
        border: 1.5px solid #EF9A9A;
        border-left: 6px solid #C62828;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .result-clickbait h2 { color: #C62828; margin: 0 0 8px; font-size: 1.35rem; }
    .result-clickbait p  { color: #B71C1C; margin: 0; font-size: 0.92rem; line-height: 1.6; }

    .result-nonclickbait {
        background: #F1FFF3;
        border: 1.5px solid #A5D6A7;
        border-left: 6px solid #2E7D32;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }
    .result-nonclickbait h2 { color: #2E7D32; margin: 0 0 8px; font-size: 1.35rem; }
    .result-nonclickbait p  { color: #1B5E20; margin: 0; font-size: 0.92rem; line-height: 1.6; }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Muat model satu kali per sesi ───────────────────────────
if "model" not in st.session_state:
    st.session_state.model   = None
    st.session_state.metrics = None

if st.session_state.model is None and model_exists():
    model, metrics = load_model()
    st.session_state.model   = model
    st.session_state.metrics = metrics


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Clickbait Detector</h1>
    <p>Deteksi judul berita <em>clickbait</em> menggunakan<br>
    Gradient Boosting + TF-IDF Character N-gram</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Informasi Model")
    st.divider()

    if st.session_state.model:
        st.success("✅ Model siap digunakan")
    else:
        st.error("❌ Model tidak ditemukan")

    st.divider()
    st.markdown("### ⚙️ Spesifikasi Model")
    st.markdown("""
| Parameter | Nilai |
|---|---|
| Algoritma | Gradient Boosting |
| Fitur | TF-IDF Char N-gram |
| ngram\_range | (3, 5) |
| min\_df | 5 |
| n\_estimators | 1000 |
| learning\_rate | 0.1 |
| max\_depth | 3 |
| random\_state | 42 |
| Split | 80% / 20% |
""")

    st.divider()
    st.markdown("### 🔄 Alur Training")
    st.markdown("""
1. Split dataset (80:20)
2. Training pada training set
3. 5-Fold CV pada training set
4. Final test pada testing set
""")

    st.divider()
    st.markdown("### 🔤 Preprocessing")
    st.markdown("1. Konversi ke huruf kecil\n2. Normalisasi spasi")

    st.divider()
    st.caption("Model dilatih di VSCode lalu di-deploy via GitHub.")


# ── Panel metrik ─────────────────────────────────────────────
if st.session_state.metrics:
    m = st.session_state.metrics

    # ── Step 3: Cross-Validation ──
    st.markdown(
        '<p class="step-label">Step 3 — 5-Fold Cross-Validation '
        '(pada Training Set)</p>',
        unsafe_allow_html=True
    )

    cv_scores = m.get("cv_scores", [])
    cv_mean   = m.get("cv_mean",   "—")
    cv_std    = m.get("cv_std",    "—")

    col_mean, col_std = st.columns(2)
    with col_mean:
        st.markdown(f"""
        <div class="cv-box">
            <div class="cv-lbl">Mean CV Accuracy</div>
            <div class="cv-val">{cv_mean}%</div>
        </div>""", unsafe_allow_html=True)
    with col_std:
        st.markdown(f"""
        <div class="cv-box">
            <div class="cv-lbl">Std Deviation</div>
            <div class="cv-val">± {cv_std}%</div>
        </div>""", unsafe_allow_html=True)

    if cv_scores:
        scores_str = "   |   ".join(
            [f"Fold {i+1}: {s}%" for i, s in enumerate(cv_scores)]
        )
        st.markdown(
            f"<p style='text-align:center;color:#607D8B;"
            f"font-size:0.79rem;font-family:monospace;margin-top:8px;'>"
            f"{scores_str}</p>",
            unsafe_allow_html=True
        )

    # ── Step 4: Final Test ──
    st.markdown(
        '<p class="step-label">Step 4 — Final Test '
        '(pada Testing Set)</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [m["accuracy"], m["precision"], m["recall"], m["f1"]],
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{val}%</div>
                <div class="lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        f"<p style='text-align:center;color:#90A4AE;font-size:0.77rem;"
        f"margin-top:6px;'>"
        f"Train: {m['train_size']:,} data  •  "
        f"Test: {m['test_size']:,} data  •  "
        f"Total: {m['total_size']:,} data</p>",
        unsafe_allow_html=True
    )
    st.divider()


# ── Input & Deteksi ──────────────────────────────────────────
if not st.session_state.model:
    st.warning(
        "⚠️ **Model tidak ditemukan.**  \n"
        "File `saved_model.pkl` dan `saved_metrics.pkl` belum ada "
        "di repository. Lakukan training via `app.py` di VSCode, "
        "kemudian upload kedua file tersebut ke GitHub."
    )
else:
    st.markdown("#### ✍️ Masukkan Judul Berita")
    st.caption("Masukkan judul berita berbahasa Inggris untuk dideteksi.")

    headline = st.text_area(
        label="input",
        placeholder='Contoh: "You Won\'t Believe What Happened Next..."',
        height=110,
        label_visibility="collapsed"
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        detect_clicked = st.button(
            "🔍  Deteksi Sekarang",
            use_container_width=True,
            type="primary"
        )
    with col_clear:
        clear_clicked = st.button("🗑  Hapus", use_container_width=True)

    if clear_clicked:
        st.rerun()

    # ── Hasil ────────────────────────────────────────────────
    if detect_clicked:
        if not headline.strip():
            st.warning("⚠️ Silakan masukkan judul berita terlebih dahulu.")
        else:
            with st.spinner("Memproses..."):
                res = predict(st.session_state.model, headline.strip())

            if res["label"] == 1:
                st.markdown(f"""
                <div class="result-clickbait">
                    <h2>⚠️ CLICKBAIT</h2>
                    <p>Tingkat keyakinan model: <strong>{res['confidence']}%</strong><br>
                    Judul ini <strong>terindikasi sebagai clickbait</strong>.
                    Kemungkinan besar dirancang untuk memancing klik
                    tanpa mencerminkan isi konten yang sebenarnya.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nonclickbait">
                    <h2>✅ NON-CLICKBAIT</h2>
                    <p>Tingkat keyakinan model: <strong>{res['confidence']}%</strong><br>
                    Judul ini <strong>tidak terindikasi sebagai clickbait</strong>.
                    Kemungkinan besar bersifat informatif dan
                    mencerminkan isi konten secara akurat.</p>
                </div>""", unsafe_allow_html=True)

            with st.expander("🔎 Detail preprocessing"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Input asli:**")
                    st.code(headline.strip(), language="text")
                with col_b:
                    st.markdown("**Setelah preprocessing:**")
                    st.code(res["text_clean"], language="text")

    # ── Contoh judul ─────────────────────────────────────────
    with st.expander("💡 Contoh judul untuk dicoba"):
        st.markdown("**Kemungkinan Clickbait:**")
        for ex in [
            "You Won't Believe What This Dog Did Next",
            "10 Shocking Secrets Doctors Don't Want You to Know",
            "This Simple Trick Will Change Your Life Forever",
            "What Happens Next Will Leave You Speechless",
        ]:
            st.code(ex, language="text")

        st.markdown("**Kemungkinan Non-Clickbait:**")
        for ex in [
            "President Signs New Climate Change Bill Into Law",
            "Scientists Discover New Species of Deep-Sea Fish",
            "Federal Reserve Raises Interest Rates by 0.25 Percent",
            "Apple Reports Record Quarterly Revenue of 90 Billion Dollars",
        ]:
            st.code(ex, language="text")
