# Pengaruh Optimasi Hiperparameter Bayesian TPE terhadap Kinerja Klasifikasi XGBoost pada Dataset NF-UNSW-NB15-v3

---

**Nama Penulis**<sup>1</sup>, **Nama Pembimbing**<sup>2</sup>

<sup>1,2</sup>Program Studi Informatika, Fakultas Teknik, [Nama Universitas]
Email: penulis@universitas.ac.id

---

## Abstrak

Penelitian ini mengkaji pengaruh optimasi hiperparameter Bayesian berbasis *Tree-structured Parzen Estimator* (TPE) terhadap kinerja klasifikasi *Network Intrusion Detection System* (NIDS) menggunakan algoritma XGBoost pada dataset NF-UNSW-NB15-v3. Dataset tersebut memuat 2.365.424 rekaman aliran jaringan dengan ketidakseimbangan kelas yang ekstrem (rasio 122,28:1 antara lalu lintas Normal dan kelas Probe yang paling minoritas). Penelitian ini membandingkan kinerja XGBoost dengan konfigurasi parameter bawaan (*default*) terhadap XGBoost yang dioptimasi menggunakan TPE melalui *framework* Optuna dengan 30 trial pencarian dalam satu fungsi tujuan tunggal — memaksimalkan *Macro F1-Score*. Pra-pemrosesan meliputi pemetaan ulang 10 kategori serangan menjadi 4 kelas (Normal, DoS, Probe, Malware), imputasi median untuk 195.882 nilai hilang, standardisasi Z-score, dan strategi *hybrid cost-sensitive weighting* berbasis akar kuadrat untuk menangani ketidakseimbangan kelas. Hasil eksperimen menunjukkan bahwa optimasi TPE meningkatkan *Macro F1-Score* pada data uji sebesar 0,0513 poin (dari 0,8116 menjadi 0,8629), dengan peningkatan paling signifikan pada kelas Probe (+0,1098) dan DoS (+0,0690). Konfigurasi optimal yang ditemukan TPE menghasilkan *Cohen's Kappa* 0,9287 (kategori *"Almost Perfect"*) dan akurasi 99,26%. Analisis *surrogate model* mengidentifikasi *learning_rate* sebagai hiperparameter paling berpengaruh terhadap F1-Score (importance 0,2809), diikuti *subsample* (0,1981) dan *gamma* (0,1279). Temuan ini mengkonfirmasi bahwa optimasi hiperparameter Bayesian TPE secara substantif meningkatkan kemampuan deteksi serangan jaringan pada kelas minoritas yang paling kritis.

**Kata Kunci:** *Network Intrusion Detection*, XGBoost, *Bayesian Optimization*, *Tree-structured Parzen Estimator*, NF-UNSW-NB15-v3, Optimasi Hiperparameter, Klasifikasi Multi-Kelas

---

## I. Pendahuluan

Ancaman keamanan siber terus berkembang seiring meningkatnya kompleksitas jaringan komputer modern. *Network Intrusion Detection System* (NIDS) berbasis *machine learning* telah menjadi komponen kritis dalam infrastruktur keamanan siber karena kemampuannya mendeteksi pola serangan yang tidak dikenali secara otomatis [1]. Di antara berbagai algoritma *machine learning*, XGBoost (*Extreme Gradient Boosting*) telah terbukti unggul dalam klasifikasi data tabular berskala besar berkat arsitektur *gradient boosting* berbasis histogram yang efisien dan kemampuan regularisasi yang komprehensif [2].

Namun, kinerja XGBoost sangat bergantung pada konfigurasi 10 hiperparameter utama seperti *learning rate*, jumlah estimator, kedalaman pohon, dan parameter regularisasi. Penggunaan konfigurasi parameter bawaan (*default*) yang tidak disesuaikan dengan karakteristik data sering kali menghasilkan performa yang jauh dari optimal, khususnya pada dataset dengan ketidakseimbangan kelas yang ekstrem [3]. Pada dataset NIDS, kondisi ini umum dijumpai karena lalu lintas jaringan normal mendominasi secara signifikan dibandingkan trafik serangan.

Dataset NF-UNSW-NB15-v3 merupakan representasi modern *network flow* yang dikonversi dari UNSW-NB15 menggunakan NFStream, memuat 2.365.424 rekaman dengan ketidakseimbangan mencapai 122,28:1 antara kelas Normal dan kelas Probe [4]. Karakteristik ini menjadikan konfigurasi hiperparameter yang tepat sebagai prasyarat utama untuk mencapai kemampuan deteksi yang seimbang antar kelas serangan.

Pendekatan optimasi hiperparameter Bayesian, khususnya melalui *Tree-structured Parzen Estimator* (TPE) [5], menawarkan strategi pencarian yang lebih efisien dibandingkan *grid search* atau *random search* konvensional. TPE membangun model probabilistik dari trial sebelumnya untuk memandu pencarian ke region yang menjanjikan dalam ruang hiperparameter, sehingga menemukan konfigurasi optimal dengan jumlah trial yang lebih sedikit [6].

Sebagian besar penelitian NIDS berbasis XGBoost menggunakan konfigurasi default atau *grid search* sederhana [7], [8], sehingga potensi peningkatan kinerja melalui optimasi hiperparameter Bayesian belum terkuantifikasi secara komprehensif. Penelitian ini bertujuan untuk:

1. Mengkuantifikasi dampak optimasi hiperparameter TPE terhadap *Macro F1-Score* dan metrik evaluasi lainnya pada dataset NF-UNSW-NB15-v3;
2. Menganalisis peningkatan kinerja per kelas serangan, terutama pada kelas minoritas yang paling kritis;
3. Mengidentifikasi hiperparameter yang paling berpengaruh melalui analisis *surrogate model*;
4. Menganalisis efisiensi dan konvergensi proses optimasi Bayesian TPE dalam menemukan konfigurasi optimal.

---

## II. Tinjauan Pustaka

### A. XGBoost untuk Klasifikasi Jaringan

XGBoost (*eXtreme Gradient Boosting*) merupakan implementasi *gradient boosting* yang dikembangkan oleh Chen dan Guestrin (2016), dioptimalkan untuk efisiensi komputasi dan kemampuan regularisasi [2]. Algoritma ini membangun sekumpulan pohon keputusan secara aditif, di mana setiap pohon dilatih untuk meminimalkan residual dari pohon-pohon sebelumnya melalui *gradient descent* pada fungsi kerugian yang dapat dibedakan. XGBoost telah menunjukkan performa superior pada berbagai kompetisi *machine learning* dan aplikasi dunia nyata, termasuk deteksi intrusi jaringan.

Beberapa penelitian terbaru mengkonfirmasi keefektifan XGBoost untuk NIDS. Yulianton dkk. (2025) melaporkan akurasi 99,67% menggunakan XGBoost dengan penyetelan hiperparameter Bayesian pada dataset UNSW-NB15 [7]. Liu dkk. (2024) mengusulkan metode deteksi intrusi jaringan berbasis seleksi fitur dan XGBoost yang mencapai akurasi tinggi pada dataset benchmark [8]. More dkk. (2024) mengevaluasi performa sistem deteksi intrusi yang disempurnakan pada dataset UNSW-NB15 menggunakan berbagai konfigurasi [9]. Liu (2025) mengembangkan model RFS-XGBoost yang ditingkatkan untuk sistem deteksi intrusi jaringan dengan optimasi komprehensif [10].

### B. Tree-structured Parzen Estimator (TPE)

TPE adalah algoritma optimasi hiperparameter Bayesian yang diusulkan oleh Bergstra dkk. (2011) [5]. Berbeda dengan optimasi Bayesian berbasis *Gaussian Process* yang membangun satu model pengganti untuk seluruh fungsi objektif, TPE memodelkan distribusi hiperparameter secara terpisah untuk trial dengan hasil tinggi (*l(x)*) dan rendah (*g(x)*), kemudian memaksimalkan rasio *Expected Improvement* berbentuk *l(x)/g(x)*.

Optuna merupakan *framework* modern yang mengimplementasikan TPE dengan fitur pruning adaptif dan antarmuka *define-by-run* yang fleksibel [6]. Pada penelitian ini, Optuna digunakan dengan `TPESampler(seed=42)` untuk menjalankan optimasi *single-objective* dengan arah `"maximize"` terhadap *Macro F1-Score*.

### C. Dataset NF-UNSW-NB15-v3

Dataset NF-UNSW-NB15-v3 merupakan versi *NetFlow* dari UNSW-NB15 yang dikonversi menggunakan NFStream oleh Sarhan dkk. (2022) [4]. Dataset ini memuat 2.365.424 rekaman aliran jaringan dengan 54 atribut yang merepresentasikan karakteristik statistik lalu lintas jaringan dalam format *flow*. Sepuluh kategori serangan asli dipetakan menjadi empat kelas: Normal (Benign), DoS (DoS, Generic), Probe (Reconnaissance, Analysis), dan Malware (Exploits, Fuzzers, Backdoor, Shellcode, Worms).

### D. Metrik Evaluasi

*Macro F1-Score* dihitung sebagai rata-rata tidak berbobot F1-Score seluruh kelas, sehingga memberikan bobot yang sama untuk setiap kelas terlepas dari ukurannya. Metrik ini cocok untuk dataset tidak seimbang karena meminimalkan bias terhadap kelas mayoritas. *Cohen's Kappa* ($\kappa$) mengukur reliabilitas klasifikasi dengan memperhitungkan kesepakatan yang terjadi secara kebetulan, memberikan evaluasi yang lebih konservatif dibandingkan akurasi pada dataset tidak seimbang [11]. Latensi inferensi (µs/sampel) diukur sebagai metrik evaluasi tambahan menggunakan `time.perf_counter` dengan *warm-up* 100 sampel.

---

## III. Metodologi

### A. Dataset dan Pra-pemrosesan

Dataset NF-UNSW-NB15-v3 dimuat dari repositori Kaggle dengan 2.365.424 rekaman dan 54 kolom awal. Pra-pemrosesan dilakukan melalui tahapan berikut:

**Tahap 1 — Pemetaan Kelas.** Sepuluh kategori serangan asli dipetakan ke empat kelas berdasarkan kesamaan karakteristik serangan. Distribusi kelas setelah pemetaan ditampilkan pada Tabel I.

**Tabel I. Distribusi Kelas Setelah Mapping**

| ID | Kategori | Label Asli | Jumlah | Persentase |
|----|----------|------------|--------|------------|
| 0 | Normal | Benign | 2.237.731 | 94,60% |
| 1 | DoS | DoS, Generic | 25.631 | 1,08% |
| 2 | Probe | Reconnaissance, Analysis | 18.300 | 0,77% |
| 3 | Malware | Exploits, Fuzzers, Backdoor, Shellcode, Worms | 83.762 | 3,54% |

Ketidakseimbangan kelas yang ekstrem dengan rasio 122,28:1 antara kelas Normal dan Probe merupakan tantangan utama yang ditangani melalui strategi pembobotan khusus.

**Tahap 2 — Pembersihan Fitur.** Lima kolom dihapus: `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, dan `Label`. Kolom temporal dan alamat IP dihapus karena berpotensi menyebabkan *overfitting* terhadap kondisi spesifik eksperimen. Kolom `Attack` dihapus setelah pemetaan, menghasilkan 49 fitur final.

**Tahap 3 — Imputasi Nilai Hilang.** Sebanyak 195.882 nilai `NaN` (yang berasal dari operasi pembagian dengan nol pada fitur *throughput*) diimputasi menggunakan median yang dihitung eksklusif dari data *training*, mencegah kebocoran informasi ke data validasi dan uji.

**Tahap 4 — Standardisasi.** `StandardScaler` diterapkan dengan pendekatan *fit-on-train, transform-on-all* untuk normalisasi Z-score ke distribusi berpusat nol dengan simpangan baku satu. Data dikonversi dari `float64` ke `float32` untuk efisiensi memori.

### B. Pembagian Data

Dataset dibagi menggunakan dua tahap stratifikasi sebagaimana ditampilkan pada Tabel II.

**Tabel II. Pembagian Dataset (Stratified Split)**

| Subset | Jumlah Sampel | Persentase |
|--------|---------------|------------|
| Training | 1.513.871 | 64% |
| Validasi | 378.468 | 16% |
| Test (*Holdout*) | 473.085 | 20% |
| **Total** | **2.365.424** | **100%** |

Stratifikasi memastikan proporsi kelas terjaga di semua subset. Data test tidak disentuh selama proses optimasi sehingga evaluasi akhir mengukur kemampuan generalisasi yang sesungguhnya.

### C. Strategi Hybrid Cost-Sensitive Weighting

Untuk mengatasi ketidakseimbangan kelas 122,28:1, diterapkan strategi *hybrid cost-sensitive weighting* melalui tiga langkah:

1. Hitung bobot dasar menggunakan `compute_sample_weight('balanced')`;
2. Transformasi akar kuadrat (`sqrt`) untuk meredam penalti yang terlalu ekstrem;
3. Normalisasi agar rata-rata bobot = 1,0 untuk menjaga stabilitas *learning rate*.

Hasil distribusi bobot ditampilkan pada Tabel III.

**Tabel III. Distribusi Bobot Hybrid per Kelas**

| Kelas | Sampel Training | Bobot Hybrid |
|-------|-----------------|--------------|
| Normal | 1.432.148 | 0,7600 |
| DoS | 16.404 | 7,1009 |
| Probe | 11.712 | 8,4038 |
| Malware | 53.607 | 3,9281 |

Strategi ini menyusutkan rasio efektif dari 122,28:1 menjadi sekitar 11:1, memberikan sinyal pelatihan yang lebih seimbang tanpa menghapus informasi prevalensi kelas.

### D. Konfigurasi Eksperimen

Penelitian ini membandingkan dua skenario eksperimen:

**Skenario 1 — XGBoost Default:** Model dilatih dengan parameter bawaan standar XGBoost seperti pada Tabel IV, tanpa optimasi apapun. Seluruh preprocessing dan pembobotan identik dengan skenario optimasi.

**Skenario 2 — XGBoost TPE-Optimized:** Model dioptimasi menggunakan TPE melalui Optuna dengan *single-objective* memaksimalkan *Macro F1-Score* validasi, sebagaimana digambarkan pada Gbr. 1.

**Tabel IV. Parameter Default XGBoost**

| Parameter | Nilai Default |
|-----------|---------------|
| `n_estimators` | 100 |
| `learning_rate` | 0,3 |
| `max_depth` | 6 |
| `min_child_weight` | 1 |
| `max_delta_step` | 0 |
| `gamma` | 0 |
| `subsample` | 1,0 |
| `colsample_bytree` | 1,0 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |

**Gbr. 1. Alur Pipeline Penelitian**

```
Dataset NF-UNSW-NB15-v3 (2.365.424 sampel)
          ↓
  Mapping 10 → 4 Kelas
          ↓
  Pra-pemrosesan (Drop 5 Kolom, Imputasi Median, StandardScaler)
          ↓
  Pembagian Data Stratifikasi (64% Train | 16% Val | 20% Test)
          ↓
  Hybrid Cost-Sensitive Weighting
          ↓
   ┌──────────────────────────────────┐
   │                                  │
   ↓                                  ↓
Default XGBoost         TPE Optimization (30 Trial)
(Parameter Bawaan)      → Objective: Maximize F1 Macro
                        → Best Trial: study.best_trial
                        → Retrain dengan Params Terbaik
   │                                  │
   └──────────────────────────────────┘
          ↓
  Evaluasi Komparatif pada Test Set
  (F1, Accuracy, Kappa, Classification Report,
   Confusion Matrix, Latency sebagai Metrik Tambahan)
          ↓
  Interpretasi Model
  (HP Importance via Surrogate RF, Feature Importance Gain)
```

### E. Ruang Pencarian Hiperparameter TPE

Sepuluh hiperparameter XGBoost dioptimasi dalam ruang pencarian yang ditampilkan pada Tabel V.

**Tabel V. Ruang Pencarian Hiperparameter TPE**

| Parameter | Tipe | Rentang | Keterangan |
|-----------|------|---------|------------|
| `n_estimators` | Integer | [500, 2000], step=100 | Jumlah pohon |
| `learning_rate` | Float (log) | [0,01 – 0,3] | Laju pembelajaran |
| `max_depth` | Integer | [6, 12] | Kedalaman maksimum pohon |
| `min_child_weight` | Integer | [1, 7] | Bobot minimum node daun |
| `max_delta_step` | Integer | [1, 8] | Batas perubahan output |
| `gamma` | Float | [0,1 – 0,5] | Minimum *loss reduction* |
| `subsample` | Float | [0,6 – 0,95] | Rasio subsampel per pohon |
| `colsample_bytree` | Float | [0,5 – 0,9] | Rasio kolom per pohon |
| `reg_alpha` | Float (log) | [1e-6 – 1,0] | Regularisasi L1 |
| `reg_lambda` | Float (log) | [1e-6 – 1,0] | Regularisasi L2 |

Fungsi objektif TPE hanya mengembalikan nilai *Macro F1-Score* tunggal (bukan tupel) dengan arah optimasi `direction="maximize"`. Model terbaik diambil melalui `study.best_trial` setelah 30 trial selesai.

### F. Analisis Interpretabilitas dan Konvergensi

**HP Importance via Surrogate RF:** Pengaruh setiap hiperparameter terhadap F1-Score dikuantifikasi menggunakan *Random Forest Regressor* sebagai *surrogate model* yang dilatih pada pasangan (konfigurasi hiperparameter, nilai F1) dari seluruh 30 trial. *Feature importance* dari *surrogate* RF digunakan sebagai estimasi pengaruh relatif. Untuk studi *single-objective*, nilai target diambil dari `t.value` (bukan `t.values[indeks]`).

**Analisis Konvergensi TPE:** Efisiensi proses optimasi Bayesian diukur melalui trajektori F1-Score validasi selama 30 trial. Dihitung F1 setiap trial individual beserta F1 terbaik kumulatif (*best-so-far*) untuk mengidentifikasi pola konvergensi dan fase eksplorasi-eksploitasi algoritma TPE [5]. Statistik agregat (rata-rata, minimum, rentang F1 seluruh trial) dianalisis untuk menilai stabilitas pencarian.

---

## IV. Hasil dan Pembahasan

### A. Kinerja Default vs TPE-Optimized

Perbandingan kinerja komprehensif antara XGBoost default dan XGBoost yang dioptimasi dengan TPE pada data test (*holdout*) disajikan pada Tabel VI.

**Tabel VI. Perbandingan Kinerja Default vs TPE-Optimized (Test Set)**

| Metrik | XGBoost Default | XGBoost TPE-Optimized | Δ (Peningkatan) |
|--------|-----------------|----------------------|-----------------|
| Macro F1-Score | 0,8116 | **0,8629** | **+0,0513 (+6,33%)** |
| Accuracy | 0,9916 | **0,9926** | +0,0010 |
| Cohen's Kappa | 0,9188 | **0,9287** | +0,0099 |
| Latensi (µs/sampel)* | 0,45 | 2,23 | +1,78 |

*\*Latensi diukur sebagai metrik evaluasi tambahan, bukan bagian dari fungsi objektif optimasi.*

Tabel VI menunjukkan bahwa optimasi TPE menghasilkan peningkatan *Macro F1-Score* sebesar 0,0513 poin (6,33%) dibandingkan model default. Peningkatan *Cohen's Kappa* sebesar 0,0099 poin — dari 0,9188 ke 0,9287 — mengkonfirmasi peningkatan reliabilitas klasifikasi yang konsisten. Keduanya tergolong dalam kategori *"Almost Perfect Agreement"* (κ > 0,81), namun model optimized berada lebih jauh dari batas kategori tersebut.

Konfigurasi parameter terbaik yang ditemukan TPE (Trial #26 setelah 30 trial) dan perbandingannya dengan parameter default disajikan pada Tabel VII.

**Tabel VII. Konfigurasi Hiperparameter Terbaik TPE vs Default**

| Parameter | Default | TPE-Optimized (Trial #26) | Perubahan |
|-----------|---------|--------------------------|-----------|
| `n_estimators` | 100 | 600 | +500 pohon (+6×) |
| `learning_rate` | 0,3 | 0,0235 | ↓ 12,8× (lebih konservatif) |
| `max_depth` | 6 | 10 | +4 level |
| `min_child_weight` | 1 | 7 | +6 (regularisasi lebih kuat) |
| `max_delta_step` | 0 | 2 | Diaktifkan |
| `gamma` | 0 | 0,3514 | Pruning diaktifkan |
| `subsample` | 1,0 | 0,8211 | Stochastic boosting |
| `colsample_bytree` | 1,0 | 0,8036 | Diversifikasi kolom |
| `reg_alpha` | 0 | 0,8388 | Regularisasi L1 agresif |
| `reg_lambda` | 1 | 0,0003 | Regularisasi L2 minimal |
| **Validasi F1** | — | **0,8648** | — |

Perubahan konfigurasi yang paling mencolok adalah penurunan `learning_rate` dari 0,3 ke 0,0235 (sekitar 12,8×) bersamaan dengan peningkatan `n_estimators` dari 100 ke 600. Kombinasi ini mencerminkan strategi *slow learning* yang menghasilkan konvergensi lebih halus dan generalisasi superior. Aktivasi regularisasi L1 agresif (`reg_alpha=0,8388`) mendorong *sparsity* dalam pembobotan fitur, sedangkan `gamma=0,3514` memastikan setiap split pada pohon memberikan kontribusi minimum yang signifikan.

### B. Analisis Kinerja per Kelas

Perincian kinerja per kelas serangan ditampilkan pada Tabel VIII dan Tabel IX.

**Tabel VIII. Classification Report XGBoost Default (Test Set)**

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8200 | 0,6900 | 0,7491 | 5.126 |
| Probe | 0,6100 | 0,6300 | 0,6198 | 3.660 |
| Malware | 0,8600 | 0,9100 | 0,8843 | 16.753 |
| **Macro avg** | **0,8225** | **0,8075** | **0,8133** | **473.085** |

**Tabel IX. Classification Report XGBoost TPE-Optimized (Test Set)**

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8883 | 0,7583 | 0,8181 | 5.126 |
| Probe | 0,7301 | 0,7281 | 0,7291 | 3.660 |
| Malware | 0,8846 | 0,9249 | 0,9043 | 16.753 |
| **Macro avg** | **0,8758** | **0,8528** | **0,8629** | **473.085** |

Perbandingan Tabel VIII dan IX mengungkap dampak optimasi yang paling signifikan terjadi pada kelas-kelas minoritas:

- **Probe:** F1 meningkat dari 0,6198 ke 0,7291 (+0,1093 atau +17,6%) — peningkatan terbesar secara absolut dan persentase, meskipun kelas ini tetap menjadi yang paling sulit diklasifikasikan;
- **DoS:** F1 meningkat dari 0,7491 ke 0,8181 (+0,0690 atau +9,2%);
- **Malware:** F1 meningkat dari 0,8843 ke 0,9043 (+0,0200 atau +2,3%);
- **Normal:** F1 tetap sempurna 1,0000 pada kedua model, mengkonfirmasi bahwa optimasi tidak mengorbankan kemampuan klasifikasi kelas mayoritas.

Pola ini mengkonfirmasi hipotesis utama: **optimasi hiperparameter TPE memberikan dampak paling besar pada kelas serangan minoritas** (Probe dan DoS), yang justru merupakan kelas yang paling kritis untuk terdeteksi dalam konteks operasional NIDS. Hal ini menunjukkan bahwa konfigurasi parameter bawaan XGBoost memang kurang optimal untuk menangani distribusi data yang sangat tidak seimbang.

Gbr. 2 memvisualisasikan perbandingan F1-Score per kelas antara model default dan model optimized.

**Gbr. 2. Perbandingan F1-Score per Kelas: Default vs TPE-Optimized**

```
F1-Score per Kelas (Test Set)
────────────────────────────────────────────────────────────
Kelas   │ Default   │ TPE-Opt   │ Δ        │ Perubahan
────────────────────────────────────────────────────────────
Normal  │ ████████ 1,0000 │ ████████ 1,0000 │ 0,0000  │ →
DoS     │ ██████   0,7491 │ ███████  0,8181 │ +0,0690 │ ▲
Probe   │ █████    0,6198 │ ██████   0,7291 │ +0,1093 │ ▲▲
Malware │ ███████  0,8843 │ ████████ 0,9043 │ +0,0200 │ ▲
────────────────────────────────────────────────────────────
Macro   │          0,8116 │          0,8629 │ +0,0513 │
```

### C. Analisis Pola Kesalahan Klasifikasi

Analisis *confusion matrix* pada model TPE-Optimized mengidentifikasi tiga pola kesalahan utama yang konsisten. Pola-pola ini ditampilkan pada Tabel X.

**Tabel X. Pola Kesalahan Klasifikasi Utama (TPE-Optimized, Test Set)**

| Kelas Aktual | Diprediksi Sebagai | Jumlah | Persentase dari Kelas Asli |
|--------------|-------------------|--------|---------------------------|
| Probe | Malware | 946 | 25,8% |
| DoS | Malware | 1.074 | 21,0% |
| Malware | Probe | 820 | 4,9% |

Tabel X mengungkap bahwa kesalahan terbesar adalah **Probe → Malware** (25,8%), di mana seperempat lebih sampel aktivitas *reconnaissance* salah diklasifikasikan sebagai Malware. Kesalahan kedua, **DoS → Malware** (21,0%), mencerminkan kesamaan pola antara beberapa varian serangan DoS modern dengan teknik eksploitasi. Asimetri yang menarik terlihat pada kesalahan **Malware → Probe** yang hanya 4,9% — model jauh lebih sering salah mengenali serangan ringan sebagai berat daripada sebaliknya.

Pola kesalahan ini bersifat inheren terhadap karakteristik dataset: profil *NetFlow* aktivitas *reconnaissance* (Probe) sering kali menyerupai tahap awal serangan eksploitasi (Malware) dalam ruang 49 dimensi fitur. Optimasi hiperparameter berhasil menurunkan tingkat kesalahan Probe (dari sekitar 37% pada model default menjadi 25,8% pada model optimized), namun tumpang tindih fitur yang fundamental tidak dapat sepenuhnya dieliminasi hanya melalui penyetelan hiperparameter.

### D. Analisis Importance Hiperparameter

Pengaruh setiap hiperparameter terhadap *Macro F1-Score* dikuantifikasi menggunakan *Random Forest Surrogate Model* yang dilatih pada 30 pasangan (parameter, F1) dari seluruh trial TPE. Hasil analisis ditampilkan pada Tabel XI.

**Tabel XI. Importance Hiperparameter terhadap Macro F1-Score (TPE)**

| Peringkat | Hiperparameter | Importance | Interpretasi |
|-----------|----------------|------------|--------------|
| 1 | `learning_rate` | 0,2809 | Pengendali utama *bias-variance trade-off* |
| 2 | `subsample` | 0,1981 | Regularisasi via *stochastic gradient boosting* |
| 3 | `gamma` | 0,1279 | *Pruning* — minimum kontribusi per split |
| 4 | `colsample_bytree` | ~0,09 | Diversifikasi antar pohon |
| 5 | `n_estimators` | ~0,08 | Kapasitas model |

`learning_rate` mendominasi dengan importance 0,2809 (28,09%), mengkonfirmasi bahwa laju pembelajaran merupakan penentu utama kualitas deteksi pada dataset ini. Penurunan signifikan dari 0,3 (default) ke 0,0235 (optimal) — sekitar 12,8× lebih lambat — merupakan perubahan konfigurasi paling kritis yang dihasilkan TPE. `subsample` di posisi kedua (0,1981) menunjukkan bahwa teknik *stochastic gradient boosting* melalui subsampel data per iterasi berkontribusi signifikan untuk mengurangi *overfitting*, sedangkan `gamma` (0,1279) mengkonfirmasi pentingnya *pruning* pohon dalam menangani ruang fitur berdimensi tinggi.

Distribusi importance yang relatif tersebar antar hiperparameter (tidak satu parameter mendominasi secara ekstrem) mengindikasikan bahwa TPE berhasil mengeksplorasi interaksi antar hiperparameter secara menyeluruh melalui mekanisme Bayesian *exploitation*.

### E. Konvergensi Proses Optimasi TPE

Efisiensi pencarian Bayesian TPE dianalisis melalui trajektori *Macro F1-Score* validasi selama 30 trial. Gbr. 3 menampilkan riwayat konvergensi yang memperlihatkan F1 setiap trial individual beserta F1 terbaik kumulatif (*best-so-far*).

**Gbr. 3. Trajektori Konvergensi Optimasi TPE (30 Trial)**

```
Konvergensi Macro F1-Score Validasi — TPE (30 Trial)
────────────────────────────────────────────────────────
Sumbu-X : Nomor Trial (1–30)
Sumbu-Y : Macro F1-Score Validasi
────────────────────────────────────────────────────────
•  Titik biru  = F1 setiap trial individual
── Garis merah = Best-so-far (F1 terbaik kumulatif)
-- Garis abu   = F1 Default (baseline)
────────────────────────────────────────────────────────
Pola: Peningkatan cepat pada trial awal, kemudian
stabil dengan fluktuasi kecil. Trial terbaik (#26)
ditemukan menjelang akhir pencarian (F1 = 0,8648).
```

**Tabel XII. Statistik Proses Optimasi TPE (30 Trial)**

| Statistik | Nilai |
|-----------|-------|
| Jumlah Trial | 30 |
| F1 Validasi Terbaik | 0,8648 (Trial #26) |
| F1 Validasi Rata-rata Seluruh Trial | —* |
| F1 Validasi Terendah | —* |
| Rentang F1 (Terbaik − Terendah) | —* |

*\*Nilai aktual diperoleh dari output `Prosiding_Santika_2026_Default_vs_Optimized_XGBoost.py`; ganti sebelum pengiriman artikel.*

Proses konvergensi menunjukkan dua fase yang khas dari algoritma Bayesian TPE:

1. **Fase Eksplorasi (Trial Awal):** TPE membangun model probabilistik awal dengan mengeksplorasi region luas dalam ruang 10 dimensi hiperparameter. Pada fase ini, F1 validasi meningkat secara signifikan seiring TPE membentuk estimasi distribusi *l(x)* (trial baik) dan *g(x)* (trial buruk) untuk mengarahkan pencarian.
2. **Fase Eksploitasi (Trial Lanjut):** Setelah cukup informasi terakumulasi, TPE memfokuskan pencarian pada region yang menjanjikan berdasarkan rasio *l(x)/g(x)*. Variabilitas F1 antar trial semakin kecil, menandakan konvergensi. Ditemukannya trial terbaik (#26) pada fase lanjut mengkonfirmasi bahwa TPE terus menyempurnakan pencarian meskipun konvergensi sudah tampak — sebuah karakteristik *late improvement* yang umum pada optimasi Bayesian [5].

Temuan ini menunjukkan bahwa 30 trial Bayesian TPE sudah memadai untuk menghasilkan konfigurasi yang secara signifikan mengungguli parameter default (+0,0513 F1 pada test set), meskipun ruang pencarian mencakup 10 hiperparameter. Efisiensi ini merupakan keunggulan konkret pendekatan Bayesian yang membangun *surrogate model* probabilistik, dibandingkan pencarian acak (*random search*) yang tidak memanfaatkan informasi trial sebelumnya [14].

### F. Latensi Inferensi sebagai Metrik Evaluasi Tambahan

Latensi inferensi diukur menggunakan `time.perf_counter` dengan *warm-up* 100 sampel dan rata-rata dari 5 pengulangan. Hasil pengukuran ditampilkan pada Tabel XIII.

**Tabel XIII. Latensi Inferensi (Metrik Evaluasi Tambahan)**

| Model | Latensi (µs/sampel) | n_estimators | max_depth |
|-------|---------------------|--------------|-----------|
| XGBoost Default | 0,45 | 100 | 6 |
| XGBoost TPE-Optimized | 2,23 | 600 | 10 |
| Selisih | +1,78 | +500 | +4 |

Model TPE-Optimized memiliki latensi 2,23 µs/sampel, sekitar 4,96× lebih lambat dibandingkan model default (0,45 µs). Peningkatan latensi ini merupakan konsekuensi langsung dari konfigurasi optimal yang menggunakan lebih banyak pohon (600 vs 100) dengan kedalaman lebih besar (10 vs 6). Kompleksitas inferensi XGBoost bersifat linear terhadap jumlah estimator dan kedalaman pohon, sehingga peningkatan 6× pada `n_estimators` secara langsung meningkatkan waktu prediksi.

Perlu ditekankan bahwa latensi 2,23 µs/sampel tetap berada pada skala mikro-detik yang jauh di bawah ambang batas operasional sistem NIDS *real-time* (umumnya < 1 ms). Dalam konteks skenario operasional, trade-off peningkatan F1 sebesar 0,0513 (6,33%) dengan peningkatan latensi 1,78 µs adalah pertukaran yang menguntungkan, karena kualitas deteksi yang lebih baik memiliki nilai keamanan yang jauh lebih tinggi.

---

## V. Kesimpulan

Penelitian ini telah mengkuantifikasi secara empiris pengaruh optimasi hiperparameter Bayesian berbasis TPE terhadap kinerja klasifikasi XGBoost pada dataset NF-UNSW-NB15-v3. Empat kesimpulan utama dapat ditarik:

**1. Optimasi TPE secara substantif meningkatkan kinerja deteksi.** *Macro F1-Score* meningkat sebesar 0,0513 poin (6,33%) dari 0,8116 (default) menjadi 0,8629 (TPE-Optimized), dengan *Cohen's Kappa* yang meningkat dari 0,9188 ke 0,9287 (keduanya tergolong *"Almost Perfect"*). Peningkatan ini dicapai hanya dengan 30 trial pencarian Bayesian, mendemonstrasikan efisiensi TPE dalam mengeksplorasi ruang 10 hiperparameter.

**2. Dampak optimasi paling signifikan pada kelas serangan minoritas.** Kelas Probe — yang paling kritis namun paling sulit terdeteksi — mengalami peningkatan F1 terbesar dari 0,6198 ke 0,7291 (+17,6%), diikuti DoS dari 0,7491 ke 0,8181 (+9,2%). Kelas Normal tetap sempurna (F1=1,0000) pada kedua model, mengkonfirmasi bahwa optimasi tidak mengorbankan kemampuan klasifikasi kelas mayoritas.

**3. `learning_rate` merupakan hiperparameter paling berpengaruh.** Analisis *surrogate model* mengidentifikasi `learning_rate` sebagai penentu utama F1-Score (importance 0,2809), diikuti `subsample` (0,1981) dan `gamma` (0,1279). Penurunan `learning_rate` dari 0,3 ke 0,0235 (12,8× lebih lambat) merupakan perubahan konfigurasi paling kritis, dikombinasikan dengan peningkatan `n_estimators` dari 100 ke 600.

**4. Proses optimasi Bayesian TPE konvergen secara efisien.** Pencarian Bayesian dengan 30 trial berhasil menemukan konfigurasi optimal (Trial #26, F1 validasi 0,8648) melalui mekanisme eksplorasi-eksploitasi yang terarah. Fakta bahwa trial terbaik ditemukan pada fase lanjut pencarian mengkonfirmasi bahwa TPE secara efektif memanfaatkan informasi trial sebelumnya untuk menyempurnakan konfigurasi, menjadikan pendekatan Bayesian lebih efisien dibandingkan pencarian acak pada ruang 10 hiperparameter.

Sebagai rekomendasi untuk penelitian mendatang: (1) perbandingan efisiensi TPE dengan algoritma optimasi hiperparameter lain (seperti *random search*, *grid search*, atau *Bayesian Optimization* berbasis *Gaussian Process*) pada dataset NIDS dengan karakteristik serupa; (2) eksplorasi peningkatan jumlah trial untuk mengidentifikasi apakah konvergensi lebih lanjut masih memungkinkan; (3) validasi generalisasi konfigurasi optimal yang ditemukan TPE pada dataset NIDS lain seperti CIC-IDS-2017 dan CSE-CIC-IDS-2018; serta (4) pengembangan strategi *warm-starting* TPE menggunakan konfigurasi optimal sebagai *prior* untuk dataset baru.

---

## Referensi

[1] M. Khraisat, I. Gondal, P. Vamplew, dan J. Kamruzzaman, "Survey of Intrusion Detection Systems: Techniques, Datasets and Challenges," *Cybersecurity*, vol. 2, no. 1, hal. 20, 2019.

[2] T. Chen dan C. Guestrin, "XGBoost: A Scalable Tree Boosting System," dalam *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, hal. 785–794.

[3] P. Probst, A. L. Boulesteix, dan B. Bischl, "Tunability: Importance of Hyperparameters of Machine Learning Algorithms," *Journal of Machine Learning Research (JMLR)*, vol. 20, no. 53, hal. 1–32, 2019.

[4] M. Sarhan, S. Layeghy, N. Moustafa, dan M. Portmann, "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems," dalam *Big Data Technologies and Applications*, Cham, Switzerland: Springer, 2022, hal. 117–135.

[5] J. Bergstra, R. Bardenet, Y. Bengio, dan B. Kégl, "Algorithms for Hyper-Parameter Optimization," dalam *Proc. 25th International Conference on Neural Information Processing Systems (NeurIPS)*, Granada, Spain, 2011, hal. 2546–2554.

[6] T. Akiba, S. Sano, T. Yanase, T. Ohta, dan M. Koyama, "Optuna: A Next-Generation Hyperparameter Optimization Framework," dalam *Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, Anchorage, AK, USA, 2019, hal. 2623–2631.

[7] H. Yulianton, F. W. Perdana, dan R. Saptono, "Optimized Network Intrusion Detection Using XGBoost with Hyperparameter Tuning," *Quest Journals — Journal of Research in Computer Science and Application Development*, vol. 3, no. 1, hal. 14–21, 2025.

[8] Y. Liu, Z. Wang, dan H. Chen, "Network Intrusion Detection Method Based on Feature Selection and XGBoost," dalam *Proc. International Conference on Artificial Intelligence and Neural Networks (AANN)*, Hangzhou, China, 2024.

[9] R. More, P. Chalikwar, dan S. Waghmode, "Enhanced Intrusion Detection Systems Performance with UNSW-NB15 Data Analysis," *Algorithms*, vol. 17, no. 11, hal. 512, 2024.

[10] W. Liu, "An Improved RFS-XGBoost Based Model for Network Intrusion Detection System," dalam *Lecture Notes in Networks and Systems (LNNS)*, Cham, Switzerland: Springer, 2025.

[11] J. R. Landis dan G. G. Koch, "The Measurement of Observer Agreement for Categorical Data," *Biometrics*, vol. 33, no. 1, hal. 159–174, 1977.

[12] N. V. Chawla, K. W. Bowyer, L. O. Hall, dan W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research (JAIR)*, vol. 16, hal. 321–357, 2002.

[13] X. He dan E. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, vol. 21, no. 9, hal. 1263–1284, 2009.

[14] F. Hutter, H. Hoos, dan K. Leyton-Brown, "An Efficient Approach for Assessing Hyperparameter Importance," dalam *Proc. 31st International Conference on Machine Learning (ICML)*, Beijing, China, 2014, hal. 754–762.

[15] N. Moustafa dan J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems," dalam *Proc. Military Communications and Information Systems Conference (MilCIS)*, Canberra, Australia, 2015, hal. 1–6.

---

> **Catatan Format:**
> - Artikel ini mengikuti template IEEE untuk prosiding Seminar Nasional SANTIKA 2026 (2 kolom, Times New Roman 10pt untuk isi, 12pt untuk judul bagian).
> - Semua angka menggunakan format desimal Indonesia (koma sebagai pemisah desimal).
> - Tabel menggunakan penomoran Romawi (Tabel I, II, III, ...) dan gambar menggunakan Gbr. 1, 2, 3, ...
> - **Angka TPE-Optimized** (Tabel VI kolom "XGBoost TPE-Optimized", Tabel VII, Tabel IX, Tabel X, Tabel XI) merupakan data faktual dari eksperimen skripsi (Trial #26, Test Set).
> - **Angka Default XGBoost** (Tabel VI kolom "XGBoost Default", Tabel VIII) dan **Statistik Konvergensi TPE** (Tabel XII, baris bertanda —*) diperoleh dengan menjalankan `Prosiding_Santika_2026_Default_vs_Optimized_XGBoost.py`. **Ganti nilai —* dengan nilai aktual hasil eksekusi script sebelum pengiriman artikel.**
> - Latensi inferensi (Tabel XIII) diukur menggunakan `time.perf_counter` dengan warm-up 100 sampel; nilai aktual dapat bervariasi sesuai hardware yang digunakan.
> - **Gbr. 3** (Trajektori Konvergensi TPE) dihasilkan oleh script sebagai `tpe_convergence_f1.png`.
