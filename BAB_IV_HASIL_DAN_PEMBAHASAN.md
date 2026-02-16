# BAB IV HASIL DAN PEMBAHASAN

---

## 4.1 Implementasi Lingkungan dan Deskripsi Data

### 4.1.1 Spesifikasi Lingkungan Eksperimen

Seluruh eksperimen dalam penelitian ini dijalankan pada platform Kaggle Notebook dengan akselerasi GPU. Konfigurasi lingkungan yang digunakan secara lengkap disajikan pada Tabel 4.1.

> **[TABEL 4.1 — Spesifikasi Lingkungan Eksperimen]**

| Komponen | Spesifikasi |
|---|---|
| Platform | Kaggle Notebook |
| GPU | NVIDIA Tesla P100-PCIE-16GB (CUDA Enabled) |
| GPU Memory | 16.384 MiB |
| Total RAM | 31,35 GB |
| RAM Tersedia (awal) | 25,53 GB (terpakai 18,6%) |
| Python | 3.12.12 |
| XGBoost | 3.1.0 (GPU support via `device='cuda'`, `tree_method='hist'`) |
| Optuna | 2.10.1 |
| NumPy | 2.0.2 |
| Pandas | 2.2.2 |
| Scikit-learn | (digunakan untuk `StandardScaler`, `LabelEncoder`, `classification_report`) |
| SciPy | (digunakan untuk uji `Kruskal-Wallis`) |
| SHAP | 0.49.1 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Random Seed | 42 |

Pemilihan GPU Tesla P100 dengan dukungan CUDA memungkinkan pelatihan XGBoost berbasis histogram (`tree_method='hist'`) secara efisien. Verifikasi GPU berhasil dilakukan pada awal eksperimen dengan output `✅ Successfully tested XGBoost GPU training`. Penggunaan `random seed = 42` diterapkan secara konsisten pada seluruh proses (pembagian data, sampler Optuna, dan pelatihan model) untuk menjamin reprodusibilitas hasil.

---

### 4.1.2 Karakteristik Dataset Terpilih

Dataset yang digunakan adalah **NF-UNSW-NB15-v3** yang bersumber dari Kaggle. Dataset ini merupakan versi NetFlow dari UNSW-NB15 yang telah dikonversi ke format aliran jaringan (network flow) menggunakan NFStream.

> **[TABEL 4.2 — Karakteristik Umum Dataset NF-UNSW-NB15-v3]**

| Parameter | Nilai |
|---|---|
| Nama Dataset | NF-UNSW-NB15-v3 |
| Sumber | Kaggle (`nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv`) |
| Total Baris (sampel) | 2.365.424 |
| Total Kolom (awal) | 54 |
| Waktu Muat | 10,92 detik |

Tabel 4.2 menyajikan ringkasan umum dataset NF-UNSW-NB15-v3 yang digunakan sebagai basis eksperimen. Dataset ini memiliki skala besar dengan 2.365.424 baris dan 54 kolom awal, yang mencakup fitur-fitur aliran jaringan berbasis NetFlow seperti durasi, ukuran paket, flag TCP, dan atribut protokol. Waktu pemuatan yang relatif singkat (10,92 detik) dimungkinkan oleh format CSV yang efisien dan kapasitas RAM yang memadai (31,35 GB).

Dataset asli memiliki 10 kategori serangan pada kolom `Attack`. Untuk menyederhanakan kompleksitas klasifikasi, dilakukan pemetaan ulang (*mapping*) menjadi **4 kelas** multi-class sesuai dengan taksonomi serangan jaringan. Distribusi kelas setelah mapping disajikan pada Tabel 4.3.

> **[TABEL 4.3 — Distribusi Kelas Setelah Mapping (4 Kelas)]**

| ID Kelas | Kategori | Label Asli yang Dipetakan | Jumlah Sampel | Persentase |
|---|---|---|---|---|
| 0 | Normal | Benign | 2.237.731 | 94,60% |
| 1 | DoS | DoS, Generic | 25.631 | 1,08% |
| 2 | Probe | Reconnaissance, Analysis | 18.300 | 0,77% |
| 3 | Malware | Exploits, Fuzzers, Backdoor, Shellcode, Worms | 83.762 | 3,54% |

> **[GAMBAR 4.1 — Diagram Batang/Pie Distribusi Kelas Dataset]**
> *Deskripsi: Visualisasi proporsi keempat kelas (Normal, DoS, Probe, Malware) yang menunjukkan ketidakseimbangan ekstrem. File referensi: `distribusi_dan_bobot_skripsi.png`.*

Gambar 4.1 memvisualisasikan distribusi keempat kelas hasil mapping secara grafis. Diagram ini menunjukkan secara jelas dominasi kelas Normal yang menempati lebih dari 94% dari keseluruhan dataset, sementara ketiga kelas serangan (DoS, Probe, Malware) secara kumulatif hanya mencakup kurang dari 6%. Visualisasi ini memperjelas skala ketidakseimbangan yang harus ditangani dalam proses pelatihan model.

Tabel 4.3 menunjukkan ketidakseimbangan kelas (*class imbalance*) yang sangat ekstrem dengan **rasio imbalance 122,28 : 1** antara kelas mayoritas (Normal = 2.237.731 sampel) dan kelas minoritas (Probe = 18.300 sampel). Kondisi ini menjadi tantangan utama yang harus diatasi melalui strategi pembobotan kelas pada tahap pelatihan model.

---

## 4.2 Hasil Pra-pemrosesan Data

### 4.2.1 Hasil Pembersihan Fitur dan Data Hilang

Tahap pertama pra-pemrosesan adalah penghapusan kolom yang tidak relevan atau berpotensi menyebabkan *data leakage* dan *overfitting*. Sebanyak **5 kolom** dihapus dari dataset:

> **[TABEL 4.4 — Kolom yang Dihapus dari Dataset]**

| No | Nama Kolom | Alasan Penghapusan |
|---|---|---|
| 1 | `FLOW_START_MILLISECONDS` | Metadata waktu, tidak relevan untuk deteksi intrusi |
| 2 | `FLOW_END_MILLISECONDS` | Metadata waktu, tidak relevan untuk deteksi intrusi |
| 3 | `IPV4_SRC_ADDR` | Identitas IP, risiko overfitting terhadap alamat spesifik |
| 4 | `IPV4_DST_ADDR` | Identitas IP, risiko overfitting terhadap alamat spesifik |
| 5 | `Label` | Label biner redundan (sudah digantikan oleh kolom `Attack` yang di-mapping) |

Tabel 4.4 merinci kelima kolom yang dieliminasi beserta justifikasi penghapusannya. Kolom `FLOW_START_MILLISECONDS` dan `FLOW_END_MILLISECONDS` dihapus karena merupakan metadata temporal yang tidak merepresentasikan karakteristik intrinsik aliran jaringan. Kolom `IPV4_SRC_ADDR` dan `IPV4_DST_ADDR` dihapus untuk mencegah model menghafal alamat IP spesifik (*overfitting*) yang tidak dapat digeneralisasi ke lingkungan jaringan lain. Kolom `Label` dihapus karena bersifat redundan — informasi klasifikasi sudah tercakup dalam kolom `Attack` yang telah dipetakan ke 4 kelas.

Setelah penghapusan kolom, dilakukan pembersihan nilai hilang (*missing values*) dan nilai tak hingga (*infinity*):

> **[TABEL 4.5 — Hasil Pembersihan Data Hilang dan Infinity]**

| Parameter | Nilai |
|---|---|
| Kolom awal (train) | 54 |
| Kolom setelah drop | 50 (49 fitur + 1 target) |
| Jumlah fitur (X) | 49 |
| NaN sebelum cleaning | 195.882 |
| NaN setelah cleaning | 0 |
| Infinity setelah cleaning | 0 |
| Metode imputasi | Median (dari data training) |
| Status | ✅ Data 100% bersih dari Infinity dan NaN |

Tabel 4.5 merangkum hasil proses pembersihan data. Dari 54 kolom awal, tersisa 50 kolom setelah penghapusan (49 fitur prediktor dan 1 kolom target). Terdapat 195.882 nilai NaN (*Not a Number*) yang terdeteksi, yang kemudian diimputasi menggunakan nilai **median** dari data training. Pemilihan median sebagai metode imputasi dilakukan karena sifatnya yang robust terhadap *outlier*, yang umum ditemukan pada data lalu lintas jaringan. Nilai tak hingga (*infinity*) yang dihasilkan dari operasi pembagian dengan nol pada beberapa fitur juga berhasil dibersihkan seluruhnya, menghasilkan dataset yang 100% bersih.

Seluruh 49 fitur yang tersisa bersifat numerik sehingga tidak diperlukan proses encoding kategorikal. Verifikasi pada output notebook mengkonfirmasi: `✅ Tidak ada fitur kategorikal (semua numerik)`.

Daftar **49 fitur** yang digunakan dalam pemodelan:

> **[TABEL 4.6 — Daftar 49 Fitur yang Digunakan]**

| No | Fitur | No | Fitur |
|---|---|---|---|
| 1 | L4_SRC_PORT | 26 | RETRANSMITTED_OUT_PKTS |
| 2 | L4_DST_PORT | 27 | SRC_TO_DST_AVG_THROUGHPUT |
| 3 | PROTOCOL | 28 | DST_TO_SRC_AVG_THROUGHPUT |
| 4 | L7_PROTO | 29 | NUM_PKTS_UP_TO_128_BYTES |
| 5 | IN_BYTES | 30 | NUM_PKTS_128_TO_256_BYTES |
| 6 | IN_PKTS | 31 | NUM_PKTS_256_TO_512_BYTES |
| 7 | OUT_BYTES | 32 | NUM_PKTS_512_TO_1024_BYTES |
| 8 | OUT_PKTS | 33 | NUM_PKTS_1024_TO_1514_BYTES |
| 9 | TCP_FLAGS | 34 | TCP_WIN_MAX_IN |
| 10 | CLIENT_TCP_FLAGS | 35 | TCP_WIN_MAX_OUT |
| 11 | SERVER_TCP_FLAGS | 36 | ICMP_TYPE |
| 12 | FLOW_DURATION_MILLISECONDS | 37 | ICMP_IPV4_TYPE |
| 13 | DURATION_IN | 38 | DNS_QUERY_ID |
| 14 | DURATION_OUT | 39 | DNS_QUERY_TYPE |
| 15 | MIN_TTL | 40 | DNS_TTL_ANSWER |
| 16 | MAX_TTL | 41 | FTP_COMMAND_RET_CODE |
| 17 | LONGEST_FLOW_PKT | 42 | SRC_TO_DST_IAT_MIN |
| 18 | SHORTEST_FLOW_PKT | 43 | SRC_TO_DST_IAT_MAX |
| 19 | MIN_IP_PKT_LEN | 44 | SRC_TO_DST_IAT_AVG |
| 20 | MAX_IP_PKT_LEN | 45 | SRC_TO_DST_IAT_STDDEV |
| 21 | SRC_TO_DST_SECOND_BYTES | 46 | DST_TO_SRC_IAT_MIN |
| 22 | DST_TO_SRC_SECOND_BYTES | 47 | DST_TO_SRC_IAT_MAX |
| 23 | RETRANSMITTED_IN_BYTES | 48 | DST_TO_SRC_IAT_AVG |
| 24 | RETRANSMITTED_IN_PKTS | 49 | DST_TO_SRC_IAT_STDDEV |
| 25 | RETRANSMITTED_OUT_BYTES | | |

Tabel 4.6 menyajikan daftar lengkap 49 fitur yang digunakan dalam pemodelan. Fitur-fitur ini mencakup beberapa kategori informasi aliran jaringan: (1) informasi port dan protokol (`L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO`), (2) volume lalu lintas (`IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`), (3) flag dan kontrol TCP (`TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`), (4) karakteristik temporal (`FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`), (5) properti paket IP (`MIN_TTL`, `MAX_TTL`, panjang paket), (6) statistik throughput dan retransmisi, (7) distribusi ukuran paket (`NUM_PKTS_UP_TO_128_BYTES` hingga `NUM_PKTS_1024_TO_1514_BYTES`), (8) informasi protokol aplikasi (DNS, FTP, ICMP), dan (9) statistik *Inter-Arrival Time* (IAT) baik untuk arah sumber-ke-tujuan maupun sebaliknya. Keragaman fitur ini memungkinkan model untuk menangkap berbagai pola serangan jaringan dari berbagai sudut pandang.


---

## 4.3 Hasil Transformasi Data

### 4.3.1 Hasil Penyandian dan Standardisasi

Pembagian dataset dilakukan secara stratifikasi (*stratified split*) untuk mempertahankan proporsi kelas di setiap subset. Dataset dibagi dalam dua tahap:

> **[TABEL 4.7 — Pembagian Dataset]**

| Subset | Jumlah Sampel | Persentase | Keterangan |
|---|---|---|---|
| Train | 1.513.871 | 64% dari total | 80% dari 80% pertama |
| Validasi | 378.468 | 16% dari total | 20% dari 80% pertama |
| Test (Holdout) | 473.085 | 20% dari total | Tidak tersentuh hingga evaluasi akhir |
| **Total** | **2.365.424** | **100%** | |

Tabel 4.7 menunjukkan strategi pembagian dataset dua tahap. Pada tahap pertama, dataset dibagi 80:20 menjadi subset latih-validasi (1.892.339 sampel) dan subset uji (473.085 sampel). Pada tahap kedua, subset latih-validasi dibagi kembali 80:20 menjadi data training final (1.513.871 sampel) dan data validasi (378.468 sampel). Subset uji (*holdout*) tidak digunakan sama sekali selama proses optimasi dan pelatihan, sehingga evaluasi akhir pada data test benar-benar mengukur kemampuan generalisasi model terhadap data yang belum pernah dilihat.

Verifikasi proporsi kelas pada data uji menunjukkan distribusi yang konsisten:
- Kelas 0 (Normal): 94,60%
- Kelas 1 (DoS): 1,08%
- Kelas 2 (Probe): 0,77%
- Kelas 3 (Malware): 3,54%

Standardisasi fitur dilakukan menggunakan **StandardScaler** dari scikit-learn dengan pendekatan *fit-on-train, transform-on-all* untuk mencegah data leakage. Seluruh 49 fitur dikonversi dari `float64` ke `float32` untuk efisiensi memori.

> **[TABEL 4.8 — Ringkasan Transformasi Data]**

| Tahap | Detail |
|---|---|
| Metode Scaling | StandardScaler (Z-score normalization) |
| Fit pada | Data Training (1.513.871 sampel) |
| Transform pada | Train, Validasi, dan Test |
| Konversi tipe data | `float64` → `float32` (hemat memori) |
| Jumlah fitur final | 49 |

Tabel 4.8 merangkum proses transformasi data yang diterapkan. StandardScaler melakukan normalisasi Z-score dengan rumus $z = (x - \mu) / \sigma$, di mana $\mu$ dan $\sigma$ dihitung hanya dari data training (1.513.871 sampel) untuk mencegah *data leakage*. Transformasi yang sama kemudian diterapkan pada data validasi dan test menggunakan parameter $\mu$ dan $\sigma$ yang telah di-*fit* dari data training. Konversi tipe data dari `float64` ke `float32` mengurangi penggunaan memori hingga 50% tanpa kehilangan presisi yang signifikan untuk keperluan klasifikasi.

---

### 4.3.2 Distribusi Bobot Penanganan Imbalance

Untuk mengatasi ketidakseimbangan kelas yang ekstrem (rasio 122,28:1), diterapkan strategi **Hybrid Cost-Sensitive Weighting** yang terdiri dari tiga langkah:

1. **Hitung bobot seimbang (*balanced*)** menggunakan `compute_sample_weight('balanced')` dari scikit-learn
2. **Transformasi akar kuadrat (`sqrt`)** untuk melunakkan penalti ekstrem pada kelas minoritas
3. **Normalisasi** agar rata-rata bobot = 1,0 (menjaga stabilitas *learning rate*)

> **[TABEL 4.9 — Distribusi Bobot Hybrid per Kelas]**

| Kelas | Jumlah Sampel (Train) | Bobot Balanced (Raw) | Bobot Hybrid (sqrt + norm) | Efektif Sampel |
|---|---|---|---|---|
| Normal | 1.432.148 (94,60%) | 0,2643 | 0,7600 | 1.088.389,79 |
| DoS | 16.404 (1,08%) | 23,0717 | 7,1009 | 116.483,76 |
| Probe | 11.712 (0,77%) | 32,3145 | 8,4038 | 98.425,14 |
| Malware | 53.607 (3,54%) | 7,0600 | 3,9281 | 210.572,31 |

Tabel 4.9 menampilkan detail distribusi bobot hybrid untuk setiap kelas. Kolom "Bobot Balanced (Raw)" menunjukkan bobot awal yang dihitung secara proporsional terbalik terhadap frekuensi kelas — kelas Probe yang paling minoritas mendapat bobot tertinggi (32,3145), sementara kelas Normal yang dominan mendapat bobot terendah (0,2643). Kolom "Bobot Hybrid" menunjukkan bobot setelah transformasi `sqrt` dan normalisasi, yang secara signifikan memoderasi perbedaan bobot (rentang 0,76–8,40 dibanding 0,26–32,31 pada bobot raw). Kolom "Efektif Sampel" menunjukkan jumlah sampel efektif setelah pembobotan — meskipun kelas Normal memiliki 1,4 juta sampel asli, efektif sampelnya turun menjadi ~1,09 juta, sementara kelas Probe naik dari ~12 ribu menjadi ~98 ribu secara efektif.

> **[GAMBAR 4.2 — Visualisasi Distribusi Bobot dan Efektif Sampel per Kelas]**
> *Deskripsi: Diagram batang yang membandingkan distribusi asli vs distribusi efektif setelah pembobotan hybrid, menunjukkan efek pemerataan kelas. File referensi: `distribusi_dan_bobot_skripsi.png`.*

Gambar 4.2 memvisualisasikan perbandingan antara jumlah sampel asli dan jumlah sampel efektif setelah pembobotan untuk setiap kelas. Diagram ini menunjukkan bagaimana strategi hybrid berhasil mempersempit kesenjangan distribusi antar kelas — rasio antara kelas terbesar dan terkecil berkurang dari 122,28:1 (distribusi asli) menjadi sekitar 11:1 (distribusi efektif). Hal ini memberikan sinyal pelatihan yang lebih seimbang kepada model tanpa sepenuhnya menghilangkan informasi prevalensi kelas.

> **[GAMBAR 4.3 — Dampak Pembobotan terhadap Distribusi Kelas]**
> *Deskripsi: Perbandingan visual antara distribusi kelas sebelum dan sesudah penerapan bobot hybrid, menunjukkan penurunan dominasi kelas Normal. File referensi: `impact_weighting_skripsi.png`.*

Gambar 4.3 menyajikan perbandingan visual *before-after* penerapan bobot hybrid terhadap distribusi kelas. Sebelum pembobotan, kelas Normal mendominasi hampir seluruh distribusi (94,60%), membuat model cenderung bias untuk selalu memprediksi Normal. Setelah pembobotan, kontribusi efektif kelas Normal berkurang secara proporsional sementara kontribusi kelas serangan meningkat, sehingga model mendapatkan insentif yang lebih kuat untuk mempelajari pola-pola kelas minoritas.

Statistik ringkasan bobot:
- **Min Weight:** 0,7600 (Normal — mayoritas, penalti rendah)
- **Max Weight:** 8,4038 (Probe — minoritas, penalti tinggi)
- **Mean Weight:** 1,0000 (ternormalisasi)

Pendekatan hybrid ini memberikan keseimbangan antara memberikan perhatian lebih pada kelas minoritas tanpa terlalu agresif yang dapat menyebabkan overfitting pada sampel minoritas.

---

## 4.4 Hasil Optimasi Model Multi-Objective

### 4.4.1 Dinamika Pencarian Hiperparameter

Optimasi hiperparameter dilakukan menggunakan pendekatan **multi-objective** dengan framework **Optuna**, di mana dua fungsi tujuan dioptimalkan secara simultan:
- **Objective 1:** Maksimasi *Macro F1-Score* (kualitas deteksi)
- **Objective 2:** Minimasi *Inference Latency* dalam µs/sample (kecepatan prediksi)

Tiga metode sampling dibandingkan, masing-masing menjalankan **30 trial**:

> **[TABEL 4.10 — Perbandingan Metode Sampling Optuna]**

| Metode | Tipe Algoritma | Konfigurasi | Jumlah Trial | Waktu Optimasi (detik) | Waktu Optimasi (menit) | Solusi Pareto |
|---|---|---|---|---|---|---|
| TPE | Bayesian (Tree-structured Parzen Estimator) | `TPESampler(seed=42)` | 30 | 1.488,16 | 24,80 | 5 |
| NSGA-II | Evolutionary Algorithm | `NSGAIISampler(seed=42, population_size=10)` | 30 | 1.883,34 | 31,39 | 4 |
| Random | Random Search (Baseline) | `RandomSampler(seed=42)` | 30 | 2.041,33 | 34,02 | 6 |

Tabel 4.10 membandingkan kinerja ketiga metode sampling selama proses optimasi. TPE merupakan metode tercepat dengan waktu optimasi 1.488,16 detik (24,80 menit), diikuti NSGA-II (31,39 menit) dan Random (34,02 menit). Meskipun lebih lambat, Random Search menghasilkan jumlah solusi Pareto terbanyak (6 solusi), menunjukkan bahwa eksplorasi acak yang luas mampu menemukan lebih banyak titik *trade-off* yang beragam. TPE yang berbasis Bayesian menghasilkan 5 solusi Pareto dengan waktu lebih efisien karena kemampuannya memodelkan hubungan antara hiperparameter dan performa, sehingga mengarahkan pencarian ke region yang lebih menjanjikan.

Total: **90 model** dilatih selama proses optimasi (~90 menit total).

Ruang pencarian hiperparameter (*search space*) yang digunakan:

> **[TABEL 4.11 — Ruang Pencarian Hiperparameter]**

| Parameter | Tipe | Rentang | Fungsi |
|---|---|---|---|
| `n_estimators` | Integer | [500, 2000], step=100 | Kompleksitas model (jumlah pohon) |
| `learning_rate` | Float (log) | [0,01 – 0,3] | Kecepatan konvergensi |
| `max_depth` | Integer | [6, 12] | Kapasitas penangkapan pola |
| `min_child_weight` | Integer | [1, 7] | Bobot minimum pada cabang |
| `max_delta_step` | Integer | [1, 8] | Stabilisasi untuk kelas imbalance |
| `gamma` | Float | [0,1 – 0,5] | Regularisasi split |
| `subsample` | Float | [0,6 – 0,95] | Sampling baris per pohon |
| `colsample_bytree` | Float | [0,5 – 0,9] | Sampling kolom per pohon |
| `reg_alpha` (L1) | Float (log) | [1e-6 – 1,0] | Seleksi fitur lunak |
| `reg_lambda` (L2) | Float (log) | [1e-6 – 1,0] | Pencegahan overfitting |

Tabel 4.11 mendefinisikan ruang pencarian 10 hiperparameter yang dioptimasi. Pemilihan rentang dan tipe sampling didasarkan pada *best practice* XGBoost: `learning_rate` dan parameter regularisasi (`reg_alpha`, `reg_lambda`) menggunakan skala logaritmik karena sensitivitas model terhadap perubahan pada orde magnitud yang berbeda. Parameter `max_delta_step` secara khusus dimasukkan karena perannya dalam menstabilkan gradien pada kasus *class imbalance*. Rentang `n_estimators` yang lebar ([500, 2000]) memungkinkan eksplorasi model dari yang ringan hingga kompleks, memberikan ruang bagi optimasi multi-objective untuk menemukan *trade-off* antara akurasi dan kecepatan.

> **[GAMBAR 4.4 — Grafik Riwayat Optimasi (Optimization History)]**
> *Deskripsi: Grafik garis yang menunjukkan dinamika pencarian per trial untuk ketiga metode, menampilkan konvergensi F1-Score dan Latency sepanjang 30 trial. File referensi: `optimization_history_final.png`.*

Gambar 4.4 menampilkan riwayat optimasi dari seluruh 30 trial untuk ketiga metode pada dua sumbu objektif (F1-Score dan Latency). Grafik ini menunjukkan pola konvergensi yang berbeda: TPE cenderung menunjukkan perbaikan bertahap yang konsisten seiring trial berjalan (*exploitation*), NSGA-II menunjukkan fluktuasi yang lebih besar karena mekanisme populasi dan mutasi (*exploration*), sementara Random Search menunjukkan sebaran yang paling acak namun sesekali menemukan solusi sangat baik. Dinamika ini mengkonfirmasi karakteristik teoritis masing-masing algoritma.

> **[TABEL 4.12 — Statistik Konvergensi Optimasi]**

| Metode | Gain F1 (%) | Gain Latency (%) | Best F1 | Best Latency (µs) |
|---|---|---|---|---|
| TPE | 1,67 | 50,01 | 0,86 | 3,70 |
| NSGA-II | 1,47 | 47,63 | 0,86 | 3,88 |
| Random | 1,82 | 47,57 | 0,87 | 3,89 |

Tabel 4.12 menunjukkan bahwa ketiga metode berhasil meningkatkan F1-Score sebesar 1,47–1,82% dan mereduksi latensi sebesar 47,57–50,01% dari trial awal ke trial terbaik. TPE mencapai peningkatan latensi tertinggi (50,01%), sementara Random mencapai peningkatan F1 tertinggi (1,82%).

---

### 4.4.2 Konfigurasi Hiperparameter Optimal

Berikut adalah konfigurasi hiperparameter dari solusi Pareto terbaik (F1 tertinggi) untuk masing-masing metode:

> **[TABEL 4.13 — Konfigurasi Hiperparameter Optimal per Metode]**

| Parameter | TPE (Trial #26) | NSGA-II (Trial #2) | Random (Trial #18) |
|---|---|---|---|
| `n_estimators` | 600 | 1.400 | 1.000 |
| `learning_rate` | 0,0235 | 0,0161 | 0,0147 |
| `max_depth` | 10 | 8 | 12 |
| `min_child_weight` | 7 | 3 | 7 |
| `max_delta_step` | 2 | 4 | 3 |
| `gamma` | 0,3514 | 0,4141 | 0,3640 |
| `subsample` | 0,8211 | 0,6699 | 0,8860 |
| `colsample_bytree` | 0,8036 | 0,7057 | 0,7221 |
| `reg_alpha` | 0,8388 | 0,0036 | 0,0015 |
| `reg_lambda` | 0,0003 | 0,0000 | 0,0000 |
| **Validasi F1** | **0,8648** | **0,8631** | **0,8660** |
| **Validasi Latency (µs)** | **2,05** | **3,84** | **4,00** |

Beberapa temuan penting dari konfigurasi optimal:
1. **Learning rate rendah** (0,0147–0,0235) konsisten di semua metode, menunjukkan bahwa konvergensi lambat menghasilkan generalisasi yang lebih baik.
2. **TPE memiliki model paling ringan** (600 pohon) dibandingkan NSGA-II (1.400) dan Random (1.000), yang menjelaskan keunggulan latensinya.
3. **Regularisasi bervariasi** — TPE mengandalkan L1 (`reg_alpha=0,8388`) sementara NSGA-II dan Random menggunakan nilai yang lebih kecil.
4. **`gamma` moderat** (0,35–0,41) pada semua metode menunjukkan kebutuhan pruning yang konsisten untuk menghindari overfitting.


---

## 4.5 Evaluasi Kinerja Model

### 4.5.1 Analisis Pareto Front (Trade-off Solusi)

Analisis Pareto front mengidentifikasi solusi-solusi yang tidak didominasi (*non-dominated*) pada ruang trade-off antara F1-Score (akurasi deteksi) dan latensi inferensi (kecepatan). Solusi Pareto-optimal adalah solusi yang tidak dapat ditingkatkan pada satu objektif tanpa menurunkan objektif lainnya.

> **[GAMBAR 4.5 — Pareto Front Statis (Ketiga Metode)]**
> *Deskripsi: Scatter plot yang menampilkan seluruh trial (titik transparan) dan solusi Pareto-optimal (titik tebal berwarna) untuk TPE (biru), NSGA-II (merah), dan Random (hijau) pada ruang F1-Score vs Latency. File referensi: `pareto_front_static_hd.png`.*

Gambar 4.5 merupakan visualisasi kunci yang menampilkan *Pareto front* dari ketiga metode pada satu bidang koordinat F1-Score (sumbu-y) vs Latensi (sumbu-x). Setiap titik merepresentasikan satu trial, dengan titik yang lebih besar dan tebal menandai solusi Pareto-optimal. Dari grafik ini terlihat bahwa solusi Pareto ketiga metode membentuk frontier yang berdekatan, mengkonfirmasi kesetaraan kualitas antar metode. TPE cenderung mendominasi region latensi rendah (kiri bawah), sementara Random menemukan solusi dengan F1 tertinggi namun latensi lebih besar (kanan atas). NSGA-II berada di antara keduanya dengan sebaran yang lebih terkontrol.

> **[TABEL 4.14 — Solusi Pareto-Optimal TPE (5 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 26 | 0,8648 | 2,05 | 600 | 0,0235 | 10 |
| 2 | 21 | 0,8633 | 1,72 | 700 | 0,0652 | 7 |
| 3 | 22 | 0,8615 | 1,62 | 700 | 0,0748 | 6 |
| 4 | 4 | 0,8594 | 1,46 | 600 | 0,0539 | 6 |
| 5 | 23 | 0,8565 | 1,40 | 500 | 0,0266 | 7 |

Tabel 4.14 menyajikan 5 solusi Pareto-optimal yang ditemukan oleh metode TPE. Solusi-solusi ini membentuk *trade-off* yang jelas: solusi #1 (Trial 26) memberikan F1 tertinggi (0,8648) namun dengan latensi terbesar (2,05 µs), sementara solusi #5 (Trial 23) memberikan latensi terendah (1,40 µs) dengan kompromi F1 yang lebih rendah (0,8565). Rentang latensi TPE (1,40–2,05 µs) secara konsisten lebih rendah dibandingkan metode lainnya, yang disebabkan oleh kemampuan TPE dalam menemukan model dengan jumlah pohon yang lebih sedikit (500–700) tanpa mengorbankan F1 secara signifikan.

> **[TABEL 4.15 — Solusi Pareto-Optimal NSGA-II (4 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 2 | 0,8631 | 3,84 | 1.400 | 0,0161 | 8 |
| 2 | 7 | 0,8601 | 3,14 | 1.700 | 0,0197 | 6 |
| 3 | 20 | 0,8597 | 3,12 | 1.700 | 0,0197 | 6 |
| 4 | 4 | 0,8594 | 1,47 | 600 | 0,0539 | 6 |

Tabel 4.15 menampilkan 4 solusi Pareto-optimal dari metode NSGA-II. Dibandingkan TPE, NSGA-II menghasilkan lebih sedikit solusi Pareto dan dengan rentang latensi yang lebih lebar (1,47–3,84 µs). Solusi terbaik NSGA-II (Trial 2, F1=0,8631) membutuhkan 1.400 pohon — lebih dari dua kali lipat jumlah pohon pada solusi terbaik TPE — yang berdampak pada latensi yang hampir dua kali lipat pula (3,84 vs 2,05 µs). Pola ini menunjukkan bahwa algoritma evolusioner NSGA-II cenderung menghasilkan model yang lebih berat untuk mencapai akurasi yang sebanding. — Solusi Pareto-Optimal Random (6 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 18 | 0,8660 | 4,00 | 1.000 | 0,0147 | 12 |
| 2 | 11 | 0,8655 | 3,64 | 900 | 0,0173 | 12 |
| 3 | 25 | 0,8641 | 3,24 | 900 | 0,0371 | 11 |
| 4 | 29 | 0,8627 | 2,02 | 500 | 0,0114 | 11 |
| 5 | 10 | 0,8621 | 1,50 | 500 | 0,0871 | 8 |
| 6 | 4 | 0,8594 | 1,47 | 600 | 0,0539 | 6 |

Tabel 4.16 menampilkan 6 solusi Pareto-optimal dari metode Random Search — jumlah terbanyak di antara ketiga metode. Solusi #1 (Trial 18) mencapai F1 tertinggi secara keseluruhan (0,8660) namun dengan latensi terbesar (4,00 µs) karena menggunakan 1.000 pohon dengan `max_depth=12`. Menariknya, dua solusi teratas Random (Trial 18 dan 11) keduanya menggunakan `max_depth=12`, menunjukkan bahwa model yang lebih dalam diperlukan untuk menangkap pola serangan yang kompleks. Meskipun Random Search tidak memiliki mekanisme pencarian terarah, keragaman eksplorasinya yang luas berhasil menemukan konfigurasi-konfigurasi yang kompetitif dan bahkan unggul dalam skor F1.

Model Pareto terbaik dari masing-masing metode kemudian dilatih ulang pada data test (holdout) untuk evaluasi akhir:

> **[TABEL 4.17 — Perbandingan Kinerja Akhir pada Data Test]**

| Metrik | NSGA-II | TPE | Random |
|---|---|---|---|
| **F1 Macro** | 0,8614 | 0,8629 | **0,8642** |
| **Accuracy** | 0,9925 | 0,9926 | **0,9927** |
| **Latency (µs/sample)** | 3,98 | **2,23** | 4,15 |
| **Training Time (s)** | 70,05 | **34,70** | 66,19 |

Tabel 4.17 merangkum perbandingan kinerja akhir dari model terbaik setiap metode pada data test (*holdout*) yang belum pernah dilihat selama pelatihan maupun optimasi. Perbedaan F1 Macro antar metode sangat kecil (0,8614–0,8642, selisih maksimum 0,0028), sementara perbedaan latensi jauh lebih mencolok — TPE hampir dua kali lebih cepat dari Random (2,23 vs 4,15 µs). Waktu pelatihan ulang (*retraining*) juga bervariasi signifikan: TPE membutuhkan hanya 34,70 detik berkat jumlah pohon yang lebih sedikit (600), sementara NSGA-II membutuhkan 70,05 detik untuk 1.400 pohon.

> **[GAMBAR 4.6 — Diagram Batang Perbandingan Metrik antar Metode]**
> *Deskripsi: Grouped bar chart yang membandingkan F1-Macro, Accuracy, Latency, dan Training Time untuk ketiga metode. File referensi: `metrics_grouped_bar.png`.*

Gambar 4.6 menyajikan perbandingan visual empat metrik kinerja utama dalam format *grouped bar chart*. Dari visualisasi ini terlihat jelas bahwa metrik F1 dan Accuracy hampir tidak dapat dibedakan secara visual antar metode (bar hampir sama tinggi), sementara perbedaan Latency dan Training Time terlihat sangat mencolok — mengkonfirmasi bahwa diferensiasi utama antar metode terletak pada efisiensi komputasi, bukan kualitas deteksi.

> **[GAMBAR 4.7 — Heatmap F1-Score per Kelas per Metode]**
> *Deskripsi: Heatmap yang menampilkan F1-Score per kelas (Normal, DoS, Probe, Malware) untuk setiap metode optimasi. File referensi: `metrics_f1_heatmap.png`.*

Gambar 4.7 menampilkan heatmap F1-Score yang memungkinkan perbandingan per-kelas secara visual. Pola warna menunjukkan bahwa kelas Normal konsisten mendapat F1 sempurna (1,0000) di semua metode (warna terdalam), kelas Malware mencapai F1 yang tinggi (~0,90), kelas DoS berada di level menengah-tinggi (~0,82), sementara kelas Probe konsisten menjadi kelas tersulit dengan F1 terendah (~0,73). Pola ini seragam di ketiga metode, menunjukkan bahwa kesulitan klasifikasi bersifat inheren terhadap karakteristik data, bukan keterbatasan metode optimasi tertentu.

> **[GAMBAR 4.8 — Scatter Plot Precision vs Recall per Kelas]**
> *Deskripsi: Scatter plot yang memposisikan setiap kelas pada ruang Precision × Recall untuk ketiga metode. File referensi: `metrics_pr_scatter.png`.*

Gambar 4.8 memvisualisasikan *trade-off* antara Precision dan Recall untuk setiap kelas pada ruang dua dimensi. Titik-titik yang berada di kanan atas menunjukkan kinerja ideal (precision dan recall sama-sama tinggi). Kelas Normal berada di pojok kanan atas (sempurna), Malware berada di kuadran tinggi dengan recall yang sedikit lebih unggul dari precision (model lebih sensitif terhadap Malware), sementara Probe dan DoS menunjukkan *trade-off* yang lebih jelas — DoS memiliki precision lebih tinggi namun recall lebih rendah, menandakan model lebih konservatif dalam mendeteksi DoS.

Hasil evaluasi menunjukkan:
- **Juara Akurasi (F1):** Random (F1 Macro = 0,8642, Accuracy = 99,27%)
- **Juara Kecepatan:** TPE (Latensi = 2,23 µs/sample, Training Time = 34,70 detik)
- Selisih F1 antar metode sangat kecil (0,0028 atau 0,28%), menunjukkan kualitas deteksi yang setara.

#### Laporan Klasifikasi Detail per Metode

> **[TABEL 4.18 — Classification Report NSGA-II (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8831 | 0,7563 | 0,8148 | 5.126 |
| Probe | 0,7301 | 0,7257 | 0,7279 | 3.660 |
| Malware | 0,8834 | 0,9235 | 0,9030 | 16.753 |
| **Macro avg** | **0,8742** | **0,8514** | **0,8614** | 473.085 |
| Weighted avg | 0,9925 | 0,9925 | 0,9925 | 473.085 |

Tabel 4.18 menyajikan laporan klasifikasi detail untuk model NSGA-II pada data test. Kelas Normal mencapai skor sempurna (precision, recall, F1 = 1,0000), menandakan bahwa model tidak pernah salah mengklasifikasikan lalu lintas normal sebagai serangan (zero false positive untuk Normal). Kelas Malware juga menunjukkan kinerja baik (F1 = 0,9030) dengan recall (0,9235) yang lebih tinggi dari precision (0,8834), artinya model lebih cenderung mendeteksi Malware meskipun kadang menghasilkan alarm palsu. Kelas DoS memiliki F1 = 0,8148 dengan recall yang lebih rendah (0,7563), menunjukkan bahwa sekitar 24% serangan DoS lolos tidak terdeteksi. Kelas Probe menjadi yang tersulit (F1 = 0,7279) karena kemiripan pola dengan kelas lain. — Classification Report TPE (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8883 | 0,7583 | 0,8181 | 5.126 |
| Probe | 0,7301 | 0,7281 | 0,7291 | 3.660 |
| Malware | 0,8846 | 0,9249 | 0,9043 | 16.753 |
| **Macro avg** | **0,8758** | **0,8528** | **0,8629** | 473.085 |
| Weighted avg | 0,9926 | 0,9926 | 0,9925 | 473.085 |

Tabel 4.19 menyajikan laporan klasifikasi untuk model TPE. Pola kinerja serupa dengan NSGA-II namun dengan sedikit peningkatan: DoS F1 meningkat menjadi 0,8181 (+0,0033), Probe F1 naik menjadi 0,7291 (+0,0012), dan Malware F1 naik menjadi 0,9043 (+0,0013). Peningkatan ini konsisten di semua kelas serangan, mengindikasikan bahwa konfigurasi hiperparameter TPE (600 pohon, `max_depth=10`) memberikan keseimbangan yang sedikit lebih baik antara kemampuan generalisasi dan kapasitas model. — Classification Report Random (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8907 | 0,7628 | 0,8218 | 5.126 |
| Probe | 0,7452 | 0,7137 | 0,7291 | 3.660 |
| Malware | 0,8829 | 0,9299 | 0,9058 | 16.753 |
| **Macro avg** | **0,8797** | **0,8516** | **0,8642** | 473.085 |
| Weighted avg | 0,9927 | 0,9927 | 0,9926 | 473.085 |

Tabel 4.20 menyajikan laporan klasifikasi untuk model Random, yang mencatat F1 Macro tertinggi (0,8642). Model ini menunjukkan precision tertinggi untuk DoS (0,8907) dan Probe (0,7452), namun recall Probe justru terendah (0,7137). Kelas Malware mencapai recall tertinggi di antara ketiga metode (0,9299), menunjukkan bahwa model Random paling sensitif terhadap serangan Malware. Perbedaan antar metode pada level per-kelas tetap sangat kecil (orde 0,001–0,003), memperkuat kesimpulan bahwa kualitas klasifikasi antar metode secara substansial setara.

> **[GAMBAR 4.9 — Ringkasan Recall per Kelas untuk Ketiga Metode]**
> *Deskripsi: Bar chart horizontal yang membandingkan recall setiap kelas serangan (DoS, Probe, Malware) antar metode. File referensi: `recall_summary_chart.png`.*

Gambar 4.9 memfokuskan perbandingan recall antar metode untuk kelas-kelas serangan (DoS, Probe, Malware). Recall dipilih sebagai metrik fokus karena dalam konteks NIDS, recall merepresentasikan *detection rate* — proporsi serangan yang berhasil dideteksi. Dari visualisasi ini terlihat bahwa Malware memiliki detection rate tertinggi (>92%) di semua metode, DoS berada di tingkat menengah (~76%), dan Probe memiliki detection rate terendah (~71–73%). Perbedaan recall antar metode untuk setiap kelas sangat minimal, mengkonfirmasi kesetaraan kemampuan deteksi.

---

### 4.5.2 Analisis Kesalahan dan Reliabilitas (Confusion Matrix & Kappa)

#### Confusion Matrix

> **[GAMBAR 4.10 — Confusion Matrix Raw (Ketiga Metode)]**
> *Deskripsi: Tiga heatmap confusion matrix berdampingan (NSGA-II, TPE, Random) yang menampilkan jumlah prediksi aktual per sel. Diagonal menunjukkan prediksi benar, off-diagonal menunjukkan kesalahan. File referensi: `cm_raw_heatmap.png`.*

Gambar 4.10 menampilkan tiga confusion matrix dalam format heatmap yang memvisualisasikan distribusi prediksi model secara absolut (jumlah sampel). Warna yang lebih gelap pada diagonal utama menunjukkan jumlah prediksi benar yang dominan. Dari visualisasi ini terlihat bahwa sel-sel off-diagonal (kesalahan) terkonsentrasi pada interaksi antara kelas DoS, Probe, dan Malware — sementara kelas Normal hampir tidak memiliki kesalahan yang terlihat secara visual karena jumlah sampelnya yang sangat besar (447.546 sampel benar).

> **[GAMBAR 4.11 — Confusion Matrix Normalized (Ketiga Metode)]**
> *Deskripsi: Tiga heatmap confusion matrix ternormalisasi (per baris/recall) yang menampilkan persentase prediksi untuk setiap kelas aktual. File referensi: `cm_norm_heatmap.png`.*

Gambar 4.11 melengkapi Gambar 4.10 dengan menyajikan confusion matrix yang dinormalisasi per baris (recall-based). Normalisasi ini penting karena menghilangkan efek perbedaan jumlah sampel antar kelas, sehingga proporsi kesalahan untuk setiap kelas dapat dibandingkan secara adil. Sel diagonal menunjukkan recall per kelas (proporsi prediksi benar), sedangkan sel off-diagonal menunjukkan proporsi kesalahan spesifik. Dari heatmap ini, pola kesalahan Probe → Malware dan DoS → Malware menjadi sangat jelas terlihat sebagai sel off-diagonal berwarna paling menonjol.

Analisis kesalahan klasifikasi (*misclassification*) mengungkap pola konsisten di ketiga metode:

> **[TABEL 4.21 — Pola Kesalahan Klasifikasi Utama]**

| Metode | Kesalahan Utama | Jumlah | % dari Kelas Asli |
|---|---|---|---|
| NSGA-II | DoS → Malware | 1.085 | 21,2% |
| NSGA-II | Probe → Malware | 957 | 26,1% |
| NSGA-II | Malware → Probe | 818 | 4,9% |
| TPE | DoS → Malware | 1.074 | 21,0% |
| TPE | Probe → Malware | 946 | 25,8% |
| TPE | Malware → Probe | 820 | 4,9% |
| Random | DoS → Malware | 1.066 | 20,8% |
| Random | Probe → Malware | 1.000 | 27,3% |
| Random | Malware → Probe | 743 | 4,4% |

Tabel 4.21 mengidentifikasi tiga pola kesalahan klasifikasi yang paling dominan secara kuantitatif. Pola pertama, **Probe → Malware** (25,8–27,3%), merupakan kesalahan terbesar di mana lebih dari seperempat sampel Probe salah diklasifikasikan sebagai Malware. Hal ini dapat disebabkan oleh tumpang tindih fitur antara aktivitas *reconnaissance* (Probe) yang sering mendahului serangan eksploitasi (Malware). Pola kedua, **DoS → Malware** (20,8–21,2%), menunjukkan bahwa sekitar seperlima serangan DoS memiliki profil lalu lintas yang menyerupai Malware. Pola ketiga, **Malware → Probe** (4,4–4,9%), memiliki tingkat kesalahan yang jauh lebih rendah namun masih konsisten di ketiga metode. Konsistensi pola kesalahan ini di seluruh metode mengindikasikan bahwa masalah bersumber dari kemiripan inheren antar kelas, bukan dari kelemahan algoritma optimasi.

Kesalahan dominan adalah **Probe → Malware** (25,8–27,3%) dan **DoS → Malware** (20,8–21,2%). Hal ini dapat dijelaskan oleh kesamaan pola lalu lintas jaringan antara kelas-kelas tersebut, di mana serangan Probe (Reconnaissance) dan DoS seringkali memiliki karakteristik flow yang mirip dengan serangan Malware (Exploits, Fuzzers).

#### Cohen's Kappa

> **[TABEL 4.22 — Skor Cohen's Kappa per Metode]**

| Metode | Kappa Score | Interpretasi |
|---|---|---|
| Random | 0,9298 | Almost Perfect (Sangat Andal) |
| TPE | 0,9287 | Almost Perfect (Sangat Andal) |
| NSGA-II | 0,9278 | Almost Perfect (Sangat Andal) |

Tabel 4.22 menampilkan skor Cohen's Kappa untuk setiap metode. Cohen's Kappa mengukur tingkat kesepakatan (*agreement*) antara prediksi model dan label aktual dengan memperhitungkan kesepakatan yang terjadi secara kebetulan. Nilai Kappa berkisar dari -1 (kesepakatan lebih buruk dari acak) hingga 1 (kesepakatan sempurna). Ketiga metode mencapai Kappa > 0,92 yang berada dalam rentang 0,81–1,00 kategori "Almost Perfect" menurut skala Landis & Koch (1977). Selisih Kappa antar metode sangat kecil (0,0020), mengkonfirmasi kesetaraan reliabilitas klasifikasi.

> **[GAMBAR 4.12 — Diagram Batang Perbandingan Cohen's Kappa]**
> *Deskripsi: Bar chart yang memvisualisasikan skor Kappa ketiga metode dengan garis threshold interpretasi (0,81–1,00 = Almost Perfect). File referensi: `kappa_comparison.png`.*

Gambar 4.12 memvisualisasikan perbandingan skor Kappa dalam format diagram batang dengan garis referensi yang menandai batas kategori interpretasi. Dari visualisasi ini terlihat bahwa ketiga batang memiliki tinggi yang hampir identik dan semuanya berada jauh di atas threshold "Almost Perfect" (0,81), memberikan konfirmasi visual bahwa seluruh model memiliki reliabilitas klasifikasi yang sangat tinggi dan konsisten.

Seluruh metode menghasilkan skor Kappa di atas 0,92, yang termasuk dalam kategori **"Almost Perfect Agreement"** menurut skala Landis & Koch (1977). Ini mengindikasikan bahwa kesepakatan antara prediksi model dan label aktual sangat tinggi dan jauh melampaui kesepakatan secara kebetulan (*chance agreement*).

---

### 4.5.3 Validasi Signifikansi Statistik (Kruskal-Wallis)

Untuk memvalidasi apakah perbedaan kinerja antar metode bersifat signifikan secara statistik, dilakukan **uji Kruskal-Wallis** (non-parametrik) pada hasil **5-Fold Stratified Cross-Validation** menggunakan 20% sampel (302.774 sampel).

> **[TABEL 4.23 — Hasil Cross-Validation 5-Fold per Metode]**

| Fold | NSGA-II F1 | NSGA-II Time (s) | TPE F1 | TPE Time (s) | Random F1 | Random Time (s) |
|---|---|---|---|---|---|---|
| 1 | 0,8490 | 0,2399 | 0,8453 | 0,1414 | 0,8454 | 0,2433 |
| 2 | 0,8487 | 0,2423 | 0,8530 | 0,1413 | 0,8523 | 0,2479 |
| 3 | 0,8400 | 0,2407 | 0,8442 | 0,1406 | 0,8426 | 0,2487 |
| 4 | 0,8400 | 0,2397 | 0,8497 | 0,1417 | 0,8435 | 0,2473 |
| 5 | 0,8443 | 0,2394 | 0,8451 | 0,1427 | 0,8435 | 0,2495 |
| **Mean** | **0,8444** | **0,2404** | **0,8475** | **0,1415** | **0,8455** | **0,2473** |

Tabel 4.23 menyajikan hasil detail 5-fold cross-validation untuk ketiga metode. Setiap fold dijalankan pada subset yang berbeda dari 20% sampel data (302.774 sampel) untuk memastikan stabilitas hasil. Dari tabel ini terlihat bahwa variasi F1 antar fold sangat kecil untuk semua metode (rentang ~0,84–0,85), menunjukkan stabilitas model yang tinggi terhadap perubahan data. TPE secara konsisten mencatat waktu inferensi yang paling cepat di setiap fold (~0,14 detik vs ~0,24 detik untuk NSGA-II dan Random), mengkonfirmasi keunggulan latensi yang teramati pada evaluasi test set.

> **[GAMBAR 4.13 — Boxplot Cross-Validation F1-Score dan Inference Time]**
> *Deskripsi: Dua boxplot berdampingan yang menampilkan distribusi F1-Score dan Inference Time dari 5-fold CV untuk ketiga metode, menunjukkan sebaran dan median. File referensi: `cv_stats_boxplot.png`.*

Gambar 4.13 memvisualisasikan distribusi hasil cross-validation dalam format boxplot. Boxplot F1-Score menunjukkan bahwa median dan *interquartile range* (IQR) ketiga metode sangat tumpang tindih, secara visual mengkonfirmasi kesetaraan performa deteksi. Sebaliknya, boxplot Inference Time menunjukkan pemisahan yang jelas — box TPE terletak jauh di bawah box NSGA-II dan Random tanpa ada tumpang tindih, memberikan bukti visual yang kuat bahwa perbedaan kecepatan bersifat konsisten dan bukan artefak pengukuran tunggal.

> **[TABEL 4.24 — Hasil Uji Kruskal-Wallis]**

| Metrik yang Diuji | H-statistic | p-value | Hasil | Interpretasi |
|---|---|---|---|---|
| **F1-Score** | 1,8600 | 0,3946 | **TIDAK Signifikan** (p > 0,05) | Performa deteksi ketiga metode dianggap **setara secara statistik** |
| **Inference Time** | 12,5000 | 0,0019 | **Signifikan** (p < 0,05) | Terdapat perbedaan nyata; **TPE adalah pemenang statistik** untuk kecepatan |

Tabel 4.24 menyajikan hasil uji Kruskal-Wallis yang merupakan uji non-parametrik untuk membandingkan distribusi tiga atau lebih kelompok independen. Uji ini dipilih karena tidak mengasumsikan normalitas distribusi data, yang sesuai dengan jumlah sampel yang relatif kecil per kelompok (5 fold). Untuk F1-Score, H-statistic yang rendah (1,86) dan p-value yang tinggi (0,3946 > 0,05) menunjukkan bahwa hipotesis nol "tidak ada perbedaan signifikan" **tidak dapat ditolak** — ketiga metode secara statistik menghasilkan F1-Score yang setara. Untuk Inference Time, H-statistic yang tinggi (12,50) dan p-value yang sangat rendah (0,0019 < 0,05) menunjukkan bahwa perbedaan waktu inferensi bersifat **signifikan secara statistik**, dengan TPE (mean = 0,1415s) sebagai pemenang yang jelas.

Temuan ini memiliki implikasi penting:
1. **Dari segi kualitas deteksi (F1-Score):** Tidak ada metode yang secara signifikan lebih unggul — ketiga metode HPO menghasilkan kualitas klasifikasi yang setara. Artinya, pemilihan metode sampling tidak berdampak signifikan pada akurasi deteksi.
2. **Dari segi kecepatan inferensi:** TPE secara signifikan lebih cepat (mean = 0,1415 detik vs NSGA-II = 0,2404 detik dan Random = 0,2473 detik), menjadikannya pilihan optimal untuk skenario *real-time* yang sensitif terhadap latensi.


---

## 4.6 Interpretasi Transparansi Model

### 4.6.1 Analisis Pengaruh Hiperparameter (Surrogate Model)

Untuk memahami hiperparameter mana yang paling berpengaruh terhadap kinerja model, dilakukan analisis menggunakan **Random Forest Surrogate Model**. Model surrogate ini dilatih untuk memprediksi metrik performa (F1-Score dan Latency) berdasarkan konfigurasi hiperparameter, kemudian *feature importance* dari surrogate model digunakan sebagai proxy untuk mengukur pengaruh relatif setiap hiperparameter.

#### Pengaruh Hiperparameter terhadap F1-Score

> **[TABEL 4.25 — Importance Hiperparameter terhadap F1-Score]**

| Peringkat | TPE (Importance) | NSGA-II (Importance) | Random (Importance) |
|---|---|---|---|
| 1 | `learning_rate` (0,2809) | `learning_rate` (0,6934) | `learning_rate` (0,6550) |
| 2 | `subsample` (0,1981) | `subsample` (0,0941) | `max_depth` (0,0739) |
| 3 | `gamma` (0,1279) | `colsample_bytree` (0,0390) | `subsample` (0,0714) |

> **[GAMBAR 4.14 — Bar Chart Importance Hiperparameter terhadap F1 (TPE)]**
> *Deskripsi: Horizontal bar chart menunjukkan kontribusi relatif setiap hiperparameter terhadap variasi F1-Score pada metode TPE. File referensi: `importance_f1_tpe.png`.*

> **[GAMBAR 4.15 — Bar Chart Importance Hiperparameter terhadap F1 (NSGA-II)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter untuk F1-Score pada metode NSGA-II. File referensi: `importance_f1_nsga-ii.png`.*

> **[GAMBAR 4.16 — Bar Chart Importance Hiperparameter terhadap F1 (Random)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter untuk F1-Score pada metode Random. File referensi: `importance_f1_random.png`.*

**Temuan:** `learning_rate` secara konsisten menjadi hiperparameter paling berpengaruh terhadap F1-Score di **ketiga metode** (importance 0,28–0,69). Hal ini menunjukkan bahwa kecepatan konvergensi (*learning rate*) merupakan faktor kritis yang paling menentukan kualitas deteksi model XGBoost. `subsample` menduduki peringkat kedua pada dua dari tiga metode, mengindikasikan bahwa teknik *bagging* (sampling baris) juga berperan penting dalam regularisasi dan generalisasi.

#### Pengaruh Hiperparameter terhadap Latency

> **[TABEL 4.26 — Importance Hiperparameter terhadap Latency]**

| Peringkat | TPE (Importance) | NSGA-II (Importance) | Random (Importance) |
|---|---|---|---|
| 1 | `n_estimators` (0,7534) | `n_estimators` (0,6570) | `n_estimators` (0,6713) |
| 2 | `reg_alpha` (0,0655) | `max_depth` (0,1343) | `max_depth` (0,1277) |
| 3 | `subsample` (0,0318) | `reg_alpha` (0,0569) | `gamma` (0,0523) |

> **[GAMBAR 4.17 — Bar Chart Importance Hiperparameter terhadap Latency (TPE)]**
> *Deskripsi: Horizontal bar chart menunjukkan kontribusi relatif setiap hiperparameter terhadap variasi latensi inferensi pada metode TPE. File referensi: `importance_time_tpe.png`.*

> **[GAMBAR 4.18 — Bar Chart Importance Hiperparameter terhadap Latency (NSGA-II)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter untuk Latency pada metode NSGA-II. File referensi: `importance_time_nsga-ii.png`.*

> **[GAMBAR 4.19 — Bar Chart Importance Hiperparameter terhadap Latency (Random)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter untuk Latency pada metode Random. File referensi: `importance_time_random.png`.*

**Temuan:** `n_estimators` (jumlah pohon) mendominasi pengaruh terhadap latensi di **ketiga metode** (importance 0,66–0,75). Ini sangat logis karena latensi inferensi berbanding lurus dengan jumlah pohon yang harus dievaluasi. `max_depth` berada di peringkat kedua pada NSGA-II dan Random, menunjukkan bahwa kedalaman pohon juga mempengaruhi waktu komputasi per prediksi.

---

### 4.6.2 Analisis Kepentingan Fitur (XGBoost Feature Importance)

Analisis kepentingan fitur (*feature importance*) menggunakan metrik **Gain** dari XGBoost, yang mengukur peningkatan rata-rata dalam kualitas split yang dihasilkan oleh setiap fitur.

> **[TABEL 4.27 — Top 10 Fitur Terpenting (Rata-rata Gain dari Ketiga Metode)]**

| Peringkat | Fitur | Importance (Gain) |
|---|---|---|
| 1 | `MIN_TTL` | 14.359,00 |
| 2 | `MAX_TTL` | 11.645,04 |
| 3 | `MIN_IP_PKT_LEN` | 3.786,21 |
| 4 | `SHORTEST_FLOW_PKT` | 2.588,01 |
| 5 | `DNS_QUERY_TYPE` | 2.484,58 |
| 6 | `MAX_IP_PKT_LEN` | 555,98 |
| 7 | `LONGEST_FLOW_PKT` | 509,29 |
| 8 | `L7_PROTO` | 377,65 |
| 9 | `DNS_TTL_ANSWER` | 316,22 |
| 10 | `L4_DST_PORT` | 311,53 |

> **[TABEL 4.28 — Top 5 Fitur Terpenting per Metode]**

| Peringkat | NSGA-II (Gain) | TPE (Gain) | Random (Gain) |
|---|---|---|---|
| 1 | MIN_TTL (3.941,30) | MIN_TTL (5.415,10) | MIN_TTL (5.002,60) |
| 2 | MAX_TTL (3.029,74) | MAX_TTL (4.595,72) | MAX_TTL (4.019,58) |
| 3 | MIN_IP_PKT_LEN (1.360,91) | SHORTEST_FLOW_PKT (1.036,94) | MIN_IP_PKT_LEN (1.444,39) |
| 4 | SHORTEST_FLOW_PKT (824,66) | MIN_IP_PKT_LEN (980,91) | DNS_QUERY_TYPE (785,55) |
| 5 | DNS_QUERY_TYPE (751,83) | DNS_QUERY_TYPE (947,20) | SHORTEST_FLOW_PKT (726,41) |

> **[GAMBAR 4.20 — Bar Chart Perbandingan Feature Importance Ketiga Metode]**
> *Deskripsi: Grouped horizontal bar chart yang menampilkan top-N fitur terpenting untuk setiap metode secara berdampingan. File referensi: `feature_importance_bar_comparison.png`.*

> **[GAMBAR 4.21 — Heatmap Feature Importance Ketiga Metode]**
> *Deskripsi: Heatmap yang menampilkan importance score untuk semua fitur di ketiga metode, memungkinkan perbandingan visual pola kepentingan fitur. File referensi: `feature_importance_heatmap_comparison.png`.*

**Interpretasi fitur-fitur terpenting:**

1. **`MIN_TTL` dan `MAX_TTL`** (Time-To-Live): Fitur paling dominan dengan gain yang jauh melampaui fitur lainnya. TTL merupakan indikator kunci karena serangan jaringan seringkali memanipulasi nilai TTL (misalnya *TTL-based evasion*) atau menghasilkan pola TTL yang berbeda dari lalu lintas normal (hop count anomali).

2. **`MIN_IP_PKT_LEN` dan `SHORTEST_FLOW_PKT`**: Ukuran paket terkecil dalam sebuah flow sangat informatif karena banyak serangan (seperti *scanning*, *reconnaissance*) menggunakan paket-paket kecil (probe packets, SYN packets).

3. **`DNS_QUERY_TYPE`**: Tipe query DNS merupakan indikator penting karena serangan seperti *DNS tunneling*, *DNS amplification*, dan *exfiltration* menggunakan tipe query yang tidak lazim.

4. **`L7_PROTO` dan `L4_DST_PORT`**: Protokol aplikasi dan port tujuan membantu membedakan jenis layanan yang menjadi target serangan.

---

## 4.7 Implementasi Simulasi Prototipe NIDS

### 4.7.1 Tampilan Antarmuka Dashboard

Prototipe NIDS diimplementasikan sebagai aplikasi web menggunakan **Streamlit** yang mensimulasikan deteksi intrusi secara *real-time*. Aplikasi ini memuat model-model Pareto-optimal yang telah diekspor dari tahap pelatihan dan menjalankan inferensi pada data simulasi paket demi paket.

> **[GAMBAR 4.22 — Tampilan Dashboard Utama NIDS (State Awal)]**
> *Deskripsi: Screenshot halaman utama dashboard yang menampilkan panel kontrol (sidebar kiri) dan area dashboard utama (kanan) dalam state awal sebelum simulasi dijalankan. Ambil screenshot dari aplikasi Streamlit yang telah di-deploy.*

**Arsitektur Aplikasi:**

Pipeline aplikasi terdiri dari 5 tahap:
1. **Pemuatan Artefak** — Model XGBoost (`.json`), `StandardScaler` (`.pkl`), dan `LabelEncoder` (`.pkl`)
2. **Akuisisi & Pra-proses** — Streaming satu baris data per iterasi, pengurutan fitur sesuai `scaler.feature_names_in_`
3. **Inferensi & Timing** — `predict_proba` dengan pengukuran latensi presisi tinggi (`time.perf_counter_ns`)
4. **Dekoding Keputusan** — Pemetaan kelas integer (0–3) ke label manusia (Normal, DoS, Probe, Malware)
5. **Visualisasi & UI** — Pembaruan indikator, gauge, kartu peringatan, dan tabel log

**Komponen Antarmuka:**

> **[TABEL 4.29 — Komponen Panel Kontrol (Sidebar)]**

| No | Komponen | Tipe Widget | Deskripsi |
|---|---|---|---|
| 1 | Metode Optimasi | `selectbox` | Pilih metode HPO: TPE, Random, atau NSGA-II |
| 2 | Model Pareto | `selectbox` | Pilih model Pareto-optimal spesifik dari katalog |
| 3 | Skenario Pengujian | `radio` | Baseline (Normal saja) vs Injection (campuran serangan) |
| 4 | Jumlah Data Simulasi | `selectbox` | 50, 100, atau 150 paket |
| 5 | Acak Data | `checkbox` | Randomisasi urutan baris per simulasi |
| 6 | Kecepatan Simulasi | `slider` | 0,5 – 5,0 detik antar paket (default: 2,0) |
| 7 | Auto-stop | `checkbox` | Jeda otomatis saat serangan terdeteksi (default: aktif) |
| 8 | Tombol START/STOP | `button` | Mulai/hentikan streaming kontinyu |
| 9 | Tombol NEXT | `button` | Maju satu paket (mode step-by-step) |
| 10 | Tombol RESET | `button` | Reset seluruh state, log, dan penghitung |

> **[TABEL 4.30 — Komponen Area Dashboard Utama]**

| No | Komponen | Deskripsi |
|---|---|---|
| 1 | Kartu Status | Indikator Normal (✅ hijau) atau Attack (🚨 merah) dengan animasi pulse |
| 2 | Metrik Latensi | Latensi inferensi saat ini dan rata-rata |
| 3 | Metrik Skenario | FP count (Baseline) atau Packet count (Injection) |
| 4 | Summary Bar | Penghitung TP / TN / FP / FN berjalan |
| 5 | Gauge Latensi | Plotly gauge (0–100 ms) dengan threshold 70 ms |
| 6 | Kartu Alert | Kartu animasi untuk TP (merah), FP (oranye), FN (ungu) |
| 7 | Tabel Log Deteksi | Tabel HTML dengan baris berwarna sesuai status deteksi (maks 150 baris) |
| 8 | Grafik Ringkasan | Pie chart (TP/TN/FP/FN), bar chart prediksi, line chart latensi (pasca-simulasi) |

**Model yang Dimuat:**

Aplikasi menyediakan **7 model Pareto-optimal** dari tiga metode:

> **[TABEL 4.31 — Katalog Model pada Dashboard]**

| Metode | File Model | F1 Score | Latency (µs) | n_estimators | max_depth |
|---|---|---|---|---|---|
| TPE | `model_tpe_pareto_1.json` | 0,8648 | 5,4 | 600 | 10 |
| TPE | `model_tpe_pareto_2.json` | 0,8633 | 4,6 | 700 | 7 |
| TPE | `model_tpe_pareto_3.json` | 0,8615 | 4,3 | 700 | 6 |
| TPE | `model_tpe_pareto_4.json` | 0,8594 | 3,8 | 600 | 6 |
| TPE | `model_tpe_pareto_5.json` | 0,8565 | 3,7 | 500 | 7 |
| Random | `model_random_pareto_6.json` | 0,8594 | 3,9 | 600 | 6 |
| NSGA-II | `model_nsga-ii_pareto_4.json` | 0,8594 | 3,9 | 600 | 6 |

Data simulasi (`simulation_data.csv`) berisi 2.000 sampel dengan distribusi: 1.896 Normal (0), 28 DoS (1), 12 Probe (2), 64 Malware (3) — merepresentasikan proporsi realistis lalu lintas jaringan.

---

### 4.7.2 Hasil Pengujian Skenario Simulasi

#### Skenario 1: Baseline (Lalu Lintas Normal)

Skenario ini hanya menggunakan data berlabel Normal (kelas 0) untuk mengukur **latensi inferensi baseline** dan mendeteksi adanya **False Positive** (alarm palsu).

> **[GAMBAR 4.23 — Screenshot Dashboard saat Skenario Baseline Berjalan]**
> *Deskripsi: Screenshot yang menampilkan dashboard dengan indikator hijau (Normal), gauge latensi, dan tabel log yang menunjukkan seluruh prediksi adalah Normal. Ambil screenshot dari aplikasi Streamlit.*

Pada skenario ini, diharapkan:
- Seluruh prediksi menghasilkan label "Normal"
- Indikator status tetap berwarna hijau (✅)
- Penghitung FP tetap 0
- Latensi inferensi stabil dan rendah

#### Skenario 2: Injection (Campuran Serangan)

Skenario ini menggunakan seluruh dataset termasuk sampel serangan untuk menguji **kemampuan deteksi** dan **responsivitas** model terhadap berbagai jenis serangan.

> **[GAMBAR 4.24 — Screenshot Dashboard saat Serangan Terdeteksi (True Positive)]**
> *Deskripsi: Screenshot yang menampilkan dashboard dengan indikator merah (🚨 Attack), kartu alert merah untuk True Positive, confidence score, dan tabel log dengan baris serangan yang ditandai merah. Ambil screenshot dari aplikasi Streamlit.*

> **[GAMBAR 4.25 — Screenshot Grafik Ringkasan Pasca-Simulasi]**
> *Deskripsi: Screenshot yang menampilkan tiga grafik ringkasan setelah simulasi selesai: (1) Pie/donut chart distribusi TP/TN/FP/FN, (2) Bar chart distribusi label prediksi, (3) Line chart latensi per paket dengan garis threshold dan rata-rata. Ambil screenshot dari aplikasi Streamlit.*

Pada skenario injection, diharapkan:
- Indikator berubah dari hijau ke merah saat paket serangan terdeteksi
- Kartu alert muncul sesuai konteks (TP merah, FP oranye, FN ungu)
- Penghitung TP meningkat untuk setiap serangan yang berhasil dideteksi
- Fitur auto-stop menghentikan simulasi saat serangan pertama terdeteksi (jika diaktifkan)
- Grafik latensi per paket menunjukkan variasi waktu inferensi sepanjang simulasi

---

## Ringkasan Daftar Gambar dan Tabel

### Daftar Tabel

| No. | Nomor | Judul Tabel |
|---|---|---|
| 1 | Tabel 4.1 | Spesifikasi Lingkungan Eksperimen |
| 2 | Tabel 4.2 | Karakteristik Umum Dataset NF-UNSW-NB15-v3 |
| 3 | Tabel 4.3 | Distribusi Kelas Setelah Mapping (4 Kelas) |
| 4 | Tabel 4.4 | Kolom yang Dihapus dari Dataset |
| 5 | Tabel 4.5 | Hasil Pembersihan Data Hilang dan Infinity |
| 6 | Tabel 4.6 | Daftar 49 Fitur yang Digunakan |
| 7 | Tabel 4.7 | Pembagian Dataset |
| 8 | Tabel 4.8 | Ringkasan Transformasi Data |
| 9 | Tabel 4.9 | Distribusi Bobot Hybrid per Kelas |
| 10 | Tabel 4.10 | Perbandingan Metode Sampling Optuna |
| 11 | Tabel 4.11 | Ruang Pencarian Hiperparameter |
| 12 | Tabel 4.12 | Statistik Konvergensi Optimasi |
| 13 | Tabel 4.13 | Konfigurasi Hiperparameter Optimal per Metode |
| 14 | Tabel 4.14 | Solusi Pareto-Optimal TPE (5 Solusi) |
| 15 | Tabel 4.15 | Solusi Pareto-Optimal NSGA-II (4 Solusi) |
| 16 | Tabel 4.16 | Solusi Pareto-Optimal Random (6 Solusi) |
| 17 | Tabel 4.17 | Perbandingan Kinerja Akhir pada Data Test |
| 18 | Tabel 4.18 | Classification Report NSGA-II (Test Set) |
| 19 | Tabel 4.19 | Classification Report TPE (Test Set) |
| 20 | Tabel 4.20 | Classification Report Random (Test Set) |
| 21 | Tabel 4.21 | Pola Kesalahan Klasifikasi Utama |
| 22 | Tabel 4.22 | Skor Cohen's Kappa per Metode |
| 23 | Tabel 4.23 | Hasil Cross-Validation 5-Fold per Metode |
| 24 | Tabel 4.24 | Hasil Uji Kruskal-Wallis |
| 25 | Tabel 4.25 | Importance Hiperparameter terhadap F1-Score |
| 26 | Tabel 4.26 | Importance Hiperparameter terhadap Latency |
| 27 | Tabel 4.27 | Top 10 Fitur Terpenting (Rata-rata Gain) |
| 28 | Tabel 4.28 | Top 5 Fitur Terpenting per Metode |
| 29 | Tabel 4.29 | Komponen Panel Kontrol (Sidebar) |
| 30 | Tabel 4.30 | Komponen Area Dashboard Utama |
| 31 | Tabel 4.31 | Katalog Model pada Dashboard |

### Daftar Gambar

| No. | Nomor | Judul Gambar | File Referensi |
|---|---|---|---|
| 1 | Gambar 4.1 | Diagram Distribusi Kelas Dataset | `distribusi_dan_bobot_skripsi.png` |
| 2 | Gambar 4.2 | Visualisasi Distribusi Bobot dan Efektif Sampel | `distribusi_dan_bobot_skripsi.png` |
| 3 | Gambar 4.3 | Dampak Pembobotan terhadap Distribusi Kelas | `impact_weighting_skripsi.png` |
| 4 | Gambar 4.4 | Grafik Riwayat Optimasi | `optimization_history_final.png` |
| 5 | Gambar 4.5 | Pareto Front Statis (Ketiga Metode) | `pareto_front_static_hd.png` |
| 6 | Gambar 4.6 | Diagram Batang Perbandingan Metrik | `metrics_grouped_bar.png` |
| 7 | Gambar 4.7 | Heatmap F1-Score per Kelas per Metode | `metrics_f1_heatmap.png` |
| 8 | Gambar 4.8 | Scatter Plot Precision vs Recall | `metrics_pr_scatter.png` |
| 9 | Gambar 4.9 | Ringkasan Recall per Kelas | `recall_summary_chart.png` |
| 10 | Gambar 4.10 | Confusion Matrix Raw | `cm_raw_heatmap.png` |
| 11 | Gambar 4.11 | Confusion Matrix Normalized | `cm_norm_heatmap.png` |
| 12 | Gambar 4.12 | Perbandingan Cohen's Kappa | `kappa_comparison.png` |
| 13 | Gambar 4.13 | Boxplot Cross-Validation | `cv_stats_boxplot.png` |
| 14 | Gambar 4.14 | Importance Hiperparameter F1 (TPE) | `importance_f1_tpe.png` |
| 15 | Gambar 4.15 | Importance Hiperparameter F1 (NSGA-II) | `importance_f1_nsga-ii.png` |
| 16 | Gambar 4.16 | Importance Hiperparameter F1 (Random) | `importance_f1_random.png` |
| 17 | Gambar 4.17 | Importance Hiperparameter Latency (TPE) | `importance_time_tpe.png` |
| 18 | Gambar 4.18 | Importance Hiperparameter Latency (NSGA-II) | `importance_time_nsga-ii.png` |
| 19 | Gambar 4.19 | Importance Hiperparameter Latency (Random) | `importance_time_random.png` |
| 20 | Gambar 4.20 | Feature Importance Bar (Perbandingan) | `feature_importance_bar_comparison.png` |
| 21 | Gambar 4.21 | Feature Importance Heatmap | `feature_importance_heatmap_comparison.png` |
| 22 | Gambar 4.22 | Dashboard Utama NIDS (State Awal) | *Screenshot dari Streamlit* |
| 23 | Gambar 4.23 | Dashboard Skenario Baseline | *Screenshot dari Streamlit* |
| 24 | Gambar 4.24 | Dashboard Serangan Terdeteksi (TP) | *Screenshot dari Streamlit* |
| 25 | Gambar 4.25 | Grafik Ringkasan Pasca-Simulasi | *Screenshot dari Streamlit* |

---

> **Catatan untuk pemindahan ke Word:**
> - Semua teks dalam blok `> **[TABEL X.X — ...]**` menandai posisi di mana tabel harus disisipkan di dokumen Word.
> - Semua teks dalam blok `> **[GAMBAR X.X — ...]**` menandai posisi di mana gambar harus disisipkan beserta caption-nya.
> - File PNG referensi tersedia di arsip output notebook (`ARSIP_LENGKAP_SKRIPSI_20260212_0748.zip`, ukuran 189,29 MB).
> - Gambar 4.22–4.25 perlu diambil secara manual sebagai screenshot dari aplikasi Streamlit yang telah di-deploy.
> - Format angka menggunakan koma desimal (Indonesia) sesuai standar penulisan skripsi.
> - Seluruh angka diambil langsung dari output eksperimen notebook tanpa pembulatan tambahan.
