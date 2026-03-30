====================================================
  CLICKBAIT DETECTOR — Panduan Lengkap
====================================================

STRUKTUR FILE
─────────────
  model_utils.py    → Fungsi inti (preprocessing, training, predict)
  app.py            → Aplikasi desktop VSCode untuk TRAINING
  streamlit_app.py  → Aplikasi web Streamlit untuk DEMO / DEPLOY
  requirements.txt  → Library untuk Streamlit Cloud
  .python-version   → Kunci Python 3.11 untuk Streamlit Cloud
  README.txt        → File ini

FILE YANG DIHASILKAN SETELAH TRAINING:
  saved_model.pkl   → Model terlatih  ← upload ke GitHub
  saved_metrics.pkl → Hasil evaluasi  ← upload ke GitHub


====================================================
  ALUR KERJA
====================================================

  [VSCode]                        [GitHub]          [Streamlit]
  app.py → latih model  →  upload .pkl ke repo  →  demo deteksi
                                  ↑
                        (lakukan sekali saja)


====================================================
  LANGKAH 1 — Instalasi (lakukan sekali)
====================================================

  Pastikan Python 3.11 terinstall, lalu:

  pip install numpy==1.26.4 pandas==2.1.4 scikit-learn==1.3.2 streamlit==1.37.0


====================================================
  LANGKAH 2 — Training Model (di VSCode)
====================================================

  python app.py

  1. Klik tombol "Latih Model dari Dataset CSV"
  2. Pilih file clickbait_data.csv
  3. Tunggu proses selesai (sekitar 10-20 menit)
  4. Setelah selesai, dua file akan muncul di folder:
       → saved_model.pkl
       → saved_metrics.pkl


====================================================
  LANGKAH 3 — Upload ke GitHub
====================================================

  Upload SEMUA file berikut ke ROOT repo GitHub:

    app.py
    streamlit_app.py
    model_utils.py
    requirements.txt
    .python-version        ← pastikan nama file tidak berubah
    saved_model.pkl        ← hasil training
    saved_metrics.pkl      ← hasil training

  Catatan: saved_model.pkl bisa berukuran 50-150 MB.
  Jika melebihi 100 MB, gunakan Git LFS:
    git lfs install
    git lfs track "*.pkl"
    git add .gitattributes


====================================================
  LANGKAH 4 — Deploy ke Streamlit Cloud
====================================================

  1. Buka https://streamlit.io/cloud
  2. New App → pilih repo GitHub kamu
  3. Main file path: streamlit_app.py
  4. Klik Deploy
  5. Selesai — Streamlit langsung membaca model dari repo,
     tidak perlu training ulang sama sekali.


====================================================
  MENJALANKAN STREAMLIT SECARA LOKAL
====================================================

  streamlit run streamlit_app.py


====================================================
  FORMAT DATASET CSV
====================================================

  Kolom yang diperlukan:

  ┌──────────────────────────────┬───────────┐
  │ headline                     │ clickbait │
  ├──────────────────────────────┼───────────┤
  │ You Won't Believe This...    │     1     │
  │ President Signs New Bill     │     0     │
  └──────────────────────────────┴───────────┘

  clickbait : 1 = clickbait, 0 = non-clickbait


====================================================
  SPESIFIKASI MODEL
====================================================

  Preprocessing  : Lowercase → Normalisasi spasi
  Fitur          : TF-IDF Character N-gram
  analyzer       : char
  ngram_range    : (3, 5)
  min_df         : 5
  Algoritma      : GradientBoostingClassifier
  n_estimators   : 1000
  learning_rate  : 0.1
  max_depth      : 3
  random_state   : 42
  Split          : 80% train / 20% test (stratified)

====================================================
