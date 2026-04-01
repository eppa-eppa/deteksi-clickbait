# ============================================================
#  app.py  —  Aplikasi Desktop (VSCode / Tkinter)
#
#  Alur training:
#    Step 1 : Split dataset (80:20)
#    Step 2 : Training pada training set
#    Step 3 : 5-Fold CV pada training set
#    Step 4 : Final test pada testing set
#
#  Jalankan: python app.py  |  Python: 3.11
# ============================================================

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from model_utils import (
    train_model, load_model, delete_model,
    model_exists, predict
)


class ClickbaitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clickbait Detector — Mode Training (VSCode)")
        self.root.geometry("760x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F2F5")

        self.model   = None
        self.metrics = None

        self._build_ui()
        self._check_model_on_startup()

    # ─────────────────────────────────────────────────────────
    def _build_ui(self):

        # ── Header ──
        header = tk.Frame(self.root, bg="#1A237E", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="🔍  Clickbait Detector",
            font=("Segoe UI", 20, "bold"),
            bg="#1A237E", fg="white"
        ).pack(side="left", padx=20, pady=15)

        self.status_dot = tk.Label(
            header, text="●", font=("Segoe UI", 14),
            bg="#1A237E", fg="#EF5350"
        )
        self.status_dot.pack(side="right", padx=5)

        self.status_label = tk.Label(
            header, text="Model belum dimuat",
            font=("Segoe UI", 10), bg="#1A237E", fg="#CFD8DC"
        )
        self.status_label.pack(side="right", padx=2)

        # ── Banner ──
        banner = tk.Frame(self.root, bg="#E8F5E9")
        banner.pack(fill="x")
        tk.Label(
            banner,
            text="💡  Mode VSCode: Latih model di sini → upload "
                 "saved_model.pkl & saved_metrics.pkl ke GitHub → deploy Streamlit.",
            font=("Segoe UI", 9), bg="#E8F5E9", fg="#2E7D32"
        ).pack(padx=16, pady=6)

        # ── Frame utama ──
        main = tk.Frame(self.root, bg="#F0F2F5")
        main.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Card input deteksi ──
        ic = self._card(main)
        ic.pack(fill="x", pady=(0, 10))

        tk.Label(
            ic, text="Coba Deteksi Judul Berita (Bahasa Inggris)",
            font=("Segoe UI", 11, "bold"),
            bg="white", fg="#263238"
        ).pack(anchor="w", padx=15, pady=(12, 4))

        self.input_text = tk.Text(
            ic, height=3, font=("Segoe UI", 11),
            bg="#F5F5F5", fg="#212121",
            relief="flat", bd=0, padx=10, pady=8, wrap="word"
        )
        self.input_text.pack(fill="x", padx=15, pady=(0, 4))
        self.input_text.bind("<Return>", self._on_enter)

        bf = tk.Frame(ic, bg="white")
        bf.pack(fill="x", padx=15, pady=(4, 10))

        self.detect_btn = tk.Button(
            bf, text="  Deteksi Sekarang  ",
            font=("Segoe UI", 11, "bold"),
            bg="#1A237E", fg="white", activebackground="#283593",
            relief="flat", cursor="hand2",
            command=self._detect, padx=8, pady=5
        )
        self.detect_btn.pack(side="left")

        tk.Button(
            bf, text="Hapus", font=("Segoe UI", 10),
            bg="#ECEFF1", fg="#546E7A", activebackground="#CFD8DC",
            relief="flat", cursor="hand2",
            command=self._clear, padx=8, pady=5
        ).pack(side="left", padx=(8, 0))

        # ── Card hasil deteksi ──
        rc = self._card(main)
        rc.pack(fill="x", pady=(0, 10))

        tk.Label(
            rc, text="Hasil Deteksi",
            font=("Segoe UI", 11, "bold"),
            bg="white", fg="#263238"
        ).pack(anchor="w", padx=15, pady=(10, 4))

        self.result_label = tk.Label(
            rc, text="— Belum ada hasil —",
            font=("Segoe UI", 16, "bold"),
            bg="white", fg="#90A4AE"
        )
        self.result_label.pack(pady=(2, 2))

        self.confidence_label = tk.Label(
            rc, text="",
            font=("Segoe UI", 9), bg="white", fg="#78909C"
        )
        self.confidence_label.pack(pady=(0, 10))

        # ── Card Step 3: CV ──
        cv_card = self._card(main)
        cv_card.pack(fill="x", pady=(0, 6))

        tk.Label(
            cv_card,
            text="Step 3  —  5-Fold Cross-Validation  (pada Training Set)",
            font=("Segoe UI", 10, "bold"),
            bg="white", fg="#1A237E"
        ).pack(anchor="w", padx=15, pady=(10, 6))

        cv_inner = tk.Frame(cv_card, bg="white")
        cv_inner.pack(fill="x", padx=15, pady=(0, 4))

        self.metric_labels = {}
        for i, (name, key) in enumerate([
            ("Mean CV Accuracy", "cv_mean"),
            ("Std Deviation",    "cv_std")
        ]):
            col = tk.Frame(cv_inner, bg="white")
            col.grid(row=0, column=i, padx=14, sticky="w")
            tk.Label(col, text=name, font=("Segoe UI", 9),
                     bg="white", fg="#78909C").pack()
            lbl = tk.Label(col, text="—",
                           font=("Segoe UI", 14, "bold"),
                           bg="white", fg="#1A237E")
            lbl.pack()
            self.metric_labels[key] = lbl

        self.cv_scores_label = tk.Label(
            cv_card, text="Skor per fold: —",
            font=("Segoe UI", 9), bg="white", fg="#607D8B"
        )
        self.cv_scores_label.pack(anchor="w", padx=15, pady=(2, 10))

        # ── Card Step 4: Final Test ──
        ft_card = self._card(main)
        ft_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            ft_card,
            text="Step 4  —  Final Test  (pada Testing Set)",
            font=("Segoe UI", 10, "bold"),
            bg="white", fg="#1A237E"
        ).pack(anchor="w", padx=15, pady=(10, 6))

        ft_inner = tk.Frame(ft_card, bg="white")
        ft_inner.pack(fill="x", padx=15, pady=(0, 10))

        for i, (name, key) in enumerate([
            ("Accuracy", "accuracy"), ("Precision", "precision"),
            ("Recall",   "recall"),   ("F1-Score",  "f1")
        ]):
            col = tk.Frame(ft_inner, bg="white")
            col.grid(row=0, column=i, padx=14, sticky="w")
            tk.Label(col, text=name, font=("Segoe UI", 9),
                     bg="white", fg="#78909C").pack()
            lbl = tk.Label(col, text="—",
                           font=("Segoe UI", 14, "bold"),
                           bg="white", fg="#1A237E")
            lbl.pack()
            self.metric_labels[key] = lbl

        # ── Tombol bawah ──
        bot = tk.Frame(self.root, bg="#F0F2F5")
        bot.pack(fill="x", padx=20, pady=(0, 14))

        tk.Button(
            bot, text="📂  Latih Model dari Dataset CSV",
            font=("Segoe UI", 10), bg="#1A237E", fg="white",
            activebackground="#283593", relief="flat", cursor="hand2",
            command=self._open_train_window, padx=10, pady=6
        ).pack(side="left")

        tk.Button(
            bot, text="🗑  Hapus Model Tersimpan",
            font=("Segoe UI", 10), bg="#FFEBEE", fg="#C62828",
            activebackground="#FFCDD2", relief="flat", cursor="hand2",
            command=self._delete_model, padx=10, pady=6
        ).pack(side="left", padx=(8, 0))

    def _card(self, parent):
        return tk.Frame(
            parent, bg="white",
            highlightbackground="#E0E0E0", highlightthickness=1
        )

    # ── Startup ──────────────────────────────────────────────
    def _check_model_on_startup(self):
        model, metrics = load_model()
        if model:
            self.model, self.metrics = model, metrics
            self._update_status(True)
            self._update_metrics_display()
        else:
            self._update_status(False)
            self.root.after(300, self._prompt_no_model)

    def _prompt_no_model(self):
        if messagebox.askyesno(
            "Model Tidak Ditemukan",
            "Model belum tersimpan.\n\n"
            "Apakah Anda ingin melatih model dari dataset CSV sekarang?"
        ):
            self._open_train_window()

    # ── Status ───────────────────────────────────────────────
    def _update_status(self, loaded: bool):
        if loaded:
            self.status_dot.config(fg="#66BB6A")
            self.status_label.config(text="Model siap digunakan")
            self.detect_btn.config(state="normal")
        else:
            self.status_dot.config(fg="#EF5350")
            self.status_label.config(text="Model belum dimuat")
            self.detect_btn.config(state="disabled")

    def _update_metrics_display(self):
        if not self.metrics:
            return
        # Step 4 — Final Test
        for key in ["accuracy", "precision", "recall", "f1"]:
            self.metric_labels[key].config(
                text=f"{self.metrics[key]}%"
            )
        # Step 3 — Cross-Validation
        self.metric_labels["cv_mean"].config(
            text=f"{self.metrics.get('cv_mean', '—')}%"
        )
        self.metric_labels["cv_std"].config(
            text=f"± {self.metrics.get('cv_std', '—')}%"
        )
        cv_scores = self.metrics.get("cv_scores", [])
        if cv_scores:
            scores_str = "  |  ".join([f"Fold {i+1}: {s}%"
                                        for i, s in enumerate(cv_scores)])
            self.cv_scores_label.config(text=f"Skor per fold: {scores_str}")

    # ── Deteksi ──────────────────────────────────────────────
    def _on_enter(self, event):
        if not event.state & 0x1:
            self._detect()
            return "break"

    def _detect(self):
        if not self.model:
            messagebox.showwarning(
                "Model Belum Dimuat",
                "Silakan latih model terlebih dahulu."
            )
            return
        raw = self.input_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("Input Kosong",
                                   "Silakan masukkan judul berita.")
            return

        res = predict(self.model, raw)
        if res["label"] == 1:
            self.result_label.config(text="⚠️   CLICKBAIT",    fg="#C62828")
            self.confidence_label.config(
                text=f"Keyakinan model: {res['confidence']}%  |  "
                     "Judul ini terindikasi sebagai clickbait.",
                fg="#E53935"
            )
        else:
            self.result_label.config(text="✅   NON-CLICKBAIT", fg="#2E7D32")
            self.confidence_label.config(
                text=f"Keyakinan model: {res['confidence']}%  |  "
                     "Judul ini tidak terindikasi sebagai clickbait.",
                fg="#388E3C"
            )

    def _clear(self):
        self.input_text.delete("1.0", "end")
        self.result_label.config(text="— Belum ada hasil —", fg="#90A4AE")
        self.confidence_label.config(text="")

    # ── Hapus model ──────────────────────────────────────────
    def _delete_model(self):
        if not model_exists():
            messagebox.showinfo("Info", "Tidak ada model tersimpan.")
            return
        if messagebox.askyesno("Konfirmasi",
                                "Model akan dihapus permanen. Lanjutkan?"):
            delete_model()
            self.model = self.metrics = None
            self._update_status(False)
            for lbl in self.metric_labels.values():
                lbl.config(text="—")
            self.cv_scores_label.config(text="Skor per fold: —")
            self.result_label.config(text="— Belum ada hasil —", fg="#90A4AE")
            self.confidence_label.config(text="")
            messagebox.showinfo("Berhasil", "Model telah dihapus.")

    # ── Window Training ──────────────────────────────────────
    def _open_train_window(self):
        win = tk.Toplevel(self.root)
        win.title("Latih Model — Clickbait Detector")
        win.geometry("540x440")
        win.resizable(False, False)
        win.configure(bg="#F0F2F5")
        win.grab_set()

        tk.Label(
            win, text="Latih Model dari Dataset CSV",
            font=("Segoe UI", 13, "bold"),
            bg="#F0F2F5", fg="#1A237E"
        ).pack(pady=(18, 4))

        tk.Label(
            win,
            text="Alur: Split → Training → 5-Fold CV (training set) → Final Test\n"
                 "Dataset CSV: kolom 'headline' dan 'clickbait' (0/1)",
            font=("Segoe UI", 9), bg="#F0F2F5",
            fg="#546E7A", justify="center"
        ).pack(pady=(0, 12))

        # Pilih file
        ff = tk.Frame(win, bg="#F0F2F5")
        ff.pack(fill="x", padx=24)

        path_var = tk.StringVar()
        tk.Entry(
            ff, textvariable=path_var, font=("Segoe UI", 10),
            state="readonly", bg="white", relief="flat",
            highlightbackground="#BDBDBD", highlightthickness=1
        ).pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 8))

        def browse():
            p = filedialog.askopenfilename(
                title="Pilih file CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if p:
                path_var.set(p)

        tk.Button(
            ff, text="Pilih File", font=("Segoe UI", 10),
            bg="#1A237E", fg="white", activebackground="#283593",
            relief="flat", cursor="hand2",
            command=browse, padx=8, pady=4
        ).pack(side="right")

        # Log box
        lf = tk.Frame(win, bg="#F0F2F5")
        lf.pack(fill="both", expand=True, padx=24, pady=12)

        log_box = tk.Text(
            lf, height=10, font=("Consolas", 9),
            bg="#263238", fg="#80CBC4",
            relief="flat", state="disabled", padx=8, pady=6
        )
        log_box.pack(fill="both", expand=True)

        def write_log(msg):
            log_box.config(state="normal")
            log_box.delete("1.0", "end")
            log_box.insert("end", msg)
            log_box.config(state="disabled")

        # Tombol
        train_btn = tk.Button(
            win, text="  Mulai Pelatihan  ",
            font=("Segoe UI", 11, "bold"),
            bg="#1A237E", fg="white", activebackground="#283593",
            relief="flat", cursor="hand2", padx=10, pady=7
        )
        train_btn.pack(pady=(0, 16))

        def on_done(success, result):
            train_btn.config(state="normal")
            if success:
                self.model, self.metrics = load_model()
                self._update_status(True)
                self._update_metrics_display()

                cv_str = "  |  ".join(
                    [f"F{i+1}:{s}%" for i, s in enumerate(result['cv_scores'])]
                )
                write_log(
                    f"✅ Semua tahapan selesai!\n\n"
                    f"── Step 3: Cross-Validation (Training Set) ──\n"
                    f"  Scores  : {cv_str}\n"
                    f"  Mean    : {result['cv_mean']}%\n"
                    f"  Std Dev : ± {result['cv_std']}%\n\n"
                    f"── Step 4: Final Test (Testing Set) ─────────\n"
                    f"  Accuracy  : {result['accuracy']}%\n"
                    f"  Precision : {result['precision']}%\n"
                    f"  Recall    : {result['recall']}%\n"
                    f"  F1-Score  : {result['f1']}%\n\n"
                    f"  Train: {result['train_size']} data  |  "
                    f"Test: {result['test_size']} data\n\n"
                    f"Upload saved_model.pkl & saved_metrics.pkl ke GitHub."
                )
                messagebox.showinfo(
                    "Training Selesai",
                    f"Step 3 — Cross-Validation\n"
                    f"  Mean CV : {result['cv_mean']}%\n"
                    f"  Std Dev : ± {result['cv_std']}%\n\n"
                    f"Step 4 — Final Test\n"
                    f"  Accuracy  : {result['accuracy']}%\n"
                    f"  Precision : {result['precision']}%\n"
                    f"  Recall    : {result['recall']}%\n"
                    f"  F1-Score  : {result['f1']}%\n\n"
                    f"Upload .pkl ke GitHub untuk Streamlit.",
                    parent=win
                )
            else:
                write_log(f"❌ Error:\n{result}")
                messagebox.showerror("Gagal", str(result), parent=win)

        def start():
            csv_path = path_var.get().strip()
            if not csv_path:
                messagebox.showwarning("Pilih File",
                                       "Pilih file CSV terlebih dahulu.",
                                       parent=win)
                return
            train_btn.config(state="disabled")
            write_log(
                "⏳ Memulai proses...\n\n"
                "  Step 1 : Split dataset (80:20)\n"
                "  Step 2 : Training model\n"
                "  Step 3 : 5-Fold CV pada training set\n"
                "  Step 4 : Final test pada testing set\n\n"
                "Estimasi waktu: 20-40 menit.\n"
                "Harap tunggu dan jangan tutup aplikasi."
            )
            threading.Thread(
                target=_run_training,
                args=(csv_path, win, write_log, on_done),
                daemon=True
            ).start()

        train_btn.config(command=start)


def _run_training(csv_path, win, write_log, on_done):
    try:
        metrics = train_model(
            csv_path,
            progress_callback=lambda msg: win.after(0, write_log, msg)
        )
        win.after(0, on_done, True, metrics)
    except Exception as e:
        win.after(0, on_done, False, str(e))


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    ClickbaitApp(root)
    root.mainloop()
