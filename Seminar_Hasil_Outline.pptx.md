# Outline PPT Seminar Hasil (Bab IV & Bab V)

> Format ini kompatibel dengan **PowerPoint Outline** (Insert → Slides from Outline). Setiap judul level-1 adalah judul slide, dan bullet di bawahnya menjadi isi slide.

# Seminar Hasil Skripsi
- Optimasi Hiperparameter Multi-Objective XGBoost
- Deteksi Intrusi Jaringan (NF-UNSW-NB15-v3)

# Outline
- Latar Belakang & Dataset
- Metodologi Eksperimen
- Hasil Optimasi & Evaluasi
- Analisis Kesalahan & Statistik
- Interpretabilitas
- Kesimpulan & Saran

# Lingkungan Eksperimen
- Platform: Kaggle Notebook
- GPU: NVIDIA Tesla P100 (16 GB)
- Python 3.12.12, XGBoost 3.1.0 (CUDA)
- Optuna 2.10.1, Scikit-learn
- Random Seed = 42

# Dataset
- NF-UNSW-NB15-v3
- Total sampel: 2.365.424
- Fitur awal: 54
- Setelah seleksi: 49 fitur + 1 target
- Kelas: Normal, DoS, Probe, Malware

# Distribusi Kelas (Imbalance)
- Normal: 94,60%
- DoS: 1,08%
- Probe: 0,77%
- Malware: 3,54%
- Tempatkan gambar: distribusi_dan_bobot_skripsi.png

# Pra-pemrosesan Data
- Hapus kolom IP & timestamp
- NaN: 195.882 → 0
- Infinity: 0
- Imputasi: median (training)

# Pembagian Data & Standardisasi
- Train: 64% (1.513.871)
- Validasi: 16% (378.468)
- Test: 20% (473.085)
- StandardScaler (fit on train)

# Pembobotan Kelas (Hybrid)
- Balanced weight → sqrt → norm
- Imbalance efektif turun ≈ 11:1
- Tempatkan gambar: Gambar 4.2 (dampak pembobotan)

# Setup Optimasi Multi-Objective
- Objective 1: Maximize Macro F1
- Objective 2: Minimize Latency (µs/sample)
- Metode: TPE, NSGA-II, Random
- 30 trial per metode (90 model)

# Riwayat Optimasi
- TPE konvergensi stabil
- Random lebih menyebar
- NSGA-II seimbang
- Tempatkan gambar: optimization_history_*.png

# Pareto Front
- Total solusi Pareto: 15
- Trade-off F1 vs Latency
- Tempatkan gambar: Gambar 4.4 (Pareto Front)

# Konfigurasi Terbaik
- TPE terbaik: F1=0,8648; Latency=2,05 µs
- Random terbaik: F1=0,8660; Latency=4,00 µs
- NSGA-II terbaik: F1=0,8631; Latency=3,84 µs

# Hasil Final Test
- F1 Macro: 0,8614–0,8642
- Accuracy: 0,9925–0,9927
- TPE paling cepat
- Tempatkan gambar: metrics_grouped_bar.png

# Heatmap F1 per Kelas
- Normal sempurna
- Malware tinggi
- DoS & Probe menengah
- Tempatkan gambar: Gambar 4.6 (Heatmap F1)

# Precision vs Recall
- Trade-off deteksi antar kelas
- Kesulitan utama pada Probe
- Tempatkan gambar: metrics_pr_scatter.png

# Confusion Matrix
- Error dominan: Probe → Malware, DoS → Malware
- Tempatkan gambar: Gambar 4.9 (raw) dan Gambar 4.10 (normalized)

# Cohen’s Kappa
- Kappa > 0,92 (Almost Perfect)
- Reliabilitas tinggi
- Tempatkan gambar: kappa_comparison.png

# Uji Statistik
- F1: p = 0,3946 (tidak signifikan)
- Latency: p = 0,0019 (signifikan)
- TPE unggul efisiensi
- Tempatkan gambar: Gambar 4.12 (Boxplot)

# Hyperparameter Importance (F1)
- learning_rate paling berpengaruh
- Tempatkan gambar: importance_f1_*.png

# Hyperparameter Importance (Latency)
- n_estimators dominan terhadap latensi
- Tempatkan gambar: importance_time_*.png

# Feature Importance
- Fitur dominan: MIN_TTL, MAX_TTL
- Konsisten di semua metode
- Tempatkan gambar: feature_importance_*.png

# Kesimpulan
- Optimasi multi-objective efektif → 15 Pareto solutions
- Kualitas deteksi setara antar metode
- TPE unggul efisiensi
- Error dominan: Probe/DoS → Malware
- TTL fitur paling dominan

# Saran
- Tambah rekayasa fitur (kurangi overlap kelas)
- Bandingkan dengan deep learning/ensemble
- Uji di dataset & traffic nyata
- Perluas budget optimasi
- Kembangkan prototipe ke sistem real-time
- Tambah interpretabilitas lokal (SHAP)
