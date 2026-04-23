# BAB IV HASIL DAN PEMBAHASAN

Bab ini memaparkan hasil eksperimen secara empiris dan sistematis, mulai dari konfigurasi lingkungan, pra-pemrosesan data, optimasi hiperparameter multi-objective, evaluasi kinerja model, hingga implementasi prototipe. Setiap temuan dianalisis berdasarkan kerangka teori yang telah diuraikan pada Bab II dan metodologi pada Bab III.

---

## 4.1 Implementasi Lingkungan dan Deskripsi Data

Tahap awal dalam setiap penelitian berbasis eksperimen komputasional adalah memastikan bahwa lingkungan eksperimen terdokumentasi secara lengkap dan dataset yang digunakan memenuhi kriteria representativitas yang memadai. Pada sub-bab ini, dipaparkan spesifikasi perangkat keras dan perangkat lunak yang digunakan selama eksperimen, serta karakteristik dataset NF-UNSW-NB15-v3 yang menjadi basis pelatihan dan evaluasi model XGBoost untuk klasifikasi intrusi jaringan. Dokumentasi yang menyeluruh terhadap kedua aspek ini merupakan prasyarat reprodusibilitas ilmiah sekaligus menjadi landasan bagi seluruh tahapan analisis yang akan disajikan pada sub-bab berikutnya.

### 4.1.1 Spesifikasi Lingkungan Eksperimen

Reprodusibilitas merupakan prinsip fundamental dalam penelitian berbasis eksperimen komputasional. Untuk memenuhi prinsip tersebut, seluruh konfigurasi perangkat keras, perangkat lunak, dan parameter penelitian didokumentasikan secara lengkap. Eksperimen dijalankan pada platform Kaggle Notebook yang menyediakan akselerasi GPU, dengan spesifikasi yang tercantum pada Tabel 4.1.

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

Tabel 4.1 merinci seluruh komponen lingkungan eksperimen. GPU Tesla P100 dipilih karena kemampuannya mengakselerasi pelatihan XGBoost berbasis histogram melalui dukungan CUDA dengan `tree_method='hist'` yang jauh lebih efisien dibandingkan pendekatan *exact greedy* pada dataset berskala besar. Keberhasilan integrasi GPU divalidasi pada awal eksperimen melalui pengujian pelatihan sederhana yang menghasilkan konfirmasi `✅ Successfully tested XGBoost GPU training`.

Penetapan `random seed = 42` diterapkan secara konsisten pada setiap tahap yang melibatkan stokastisitas — pembagian data, inisialisasi sampler Optuna, dan pelatihan model — guna menjamin bahwa seluruh hasil dapat direproduksi secara identik pada percobaan ulang.

Kode Program 4.1 menampilkan potongan kode konfigurasi lingkungan eksperimen dan verifikasi ketersediaan GPU yang dijalankan pada awal notebook.

> **[KODE PROGRAM 4.1 — Konfigurasi Lingkungan & Verifikasi GPU]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 1*

```
 1  import xgboost as xgb
 2  from sklearn.model_selection import train_test_split, StratifiedKFold
 3  from sklearn.preprocessing import LabelEncoder, StandardScaler
 4  from sklearn.metrics import (classification_report, confusion_matrix,
 5                               f1_score, accuracy_score, cohen_kappa_score)
 6  from sklearn.utils.class_weight import compute_sample_weight
 7  import optuna
 8  from optuna.samplers import TPESampler, NSGAIISampler, RandomSampler
 9  from scipy.stats import kruskal
10
11  RANDOM_SEED = 42
12  np.random.seed(RANDOM_SEED)
13
14  test_params = {'tree_method': 'hist', 'device': 'cuda'}
15  test_model = xgb.XGBClassifier(**test_params, n_estimators=1, random_state=42)
16  X_test = np.random.rand(100, 10)
17  y_test = np.random.randint(0, 2, 100)
18  test_model.fit(X_test, y_test, verbose=False)
```

Kode Program 4.1 menunjukkan bahwa seluruh pustaka inti penelitian berhasil diinisialisasi dengan baik, seed acak telah ditetapkan untuk menjaga reprodusibilitas, dan eksekusi pelatihan singkat pada `XGBClassifier` berbasis CUDA memvalidasi bahwa akselerasi GPU aktif sebelum eksperimen utama dijalankan.

---

### 4.1.2 Karakteristik Dataset Terpilih

Pemilihan dataset yang representatif merupakan faktor krusial dalam evaluasi sistem deteksi intrusi jaringan. Penelitian ini menggunakan dataset **NF-UNSW-NB15-v3** yang bersumber dari repositori Kaggle. Dataset ini merupakan versi NetFlow dari UNSW-NB15 yang telah dikonversi ke format aliran jaringan menggunakan NFStream, sehingga lebih merepresentasikan kondisi monitoring jaringan modern berbasis *flow* dibandingkan format paket mentah. Ringkasan karakteristik dataset disajikan pada Tabel 4.2.

> **[TABEL 4.2 — Karakteristik Umum Dataset NF-UNSW-NB15-v3]**

| Parameter | Nilai |
|---|---|
| Nama Dataset | NF-UNSW-NB15-v3 |
| Sumber | Kaggle (`nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv`) |
| Total Baris (sampel) | 2.365.424 |
| Total Kolom (awal) | 54 |
| Waktu Muat | 10,92 detik |

Tabel 4.2 merangkum profil umum dataset yang mencakup 2.365.424 record aliran jaringan dengan 54 atribut. Skala dataset yang melebihi dua juta sampel memberikan basis statistik yang kuat untuk pelatihan dan evaluasi model klasifikasi multi-kelas, sekaligus menguji skalabilitas algoritma terhadap data berdimensi tinggi.

Dataset asli memuat 10 kategori serangan pada kolom `Attack`. Untuk menyederhanakan kompleksitas klasifikasi sekaligus mempertahankan relevansi taksonomi ancaman jaringan, dilakukan pemetaan ulang menjadi **4 kelas** sesuai pengelompokan jenis serangan yang diuraikan pada Bab III. Distribusi kelas setelah pemetaan ditampilkan pada Tabel 4.3.

> **[TABEL 4.3 — Distribusi Kelas Setelah Mapping (4 Kelas)]**

| ID Kelas | Kategori | Label Asli yang Dipetakan | Jumlah Sampel | Persentase |
|---|---|---|---|---|
| 0 | Normal | Benign | 2.237.731 | 94,60% |
| 1 | DoS | DoS, Generic | 25.631 | 1,08% |
| 2 | Probe | Reconnaissance, Analysis | 18.300 | 0,77% |
| 3 | Malware | Exploits, Fuzzers, Backdoor, Shellcode, Worms | 83.762 | 3,54% |

Tabel 4.3 mengungkap ketidakseimbangan kelas yang sangat ekstrem: kelas Normal mendominasi 94,60% dari total sampel, sedangkan ketiga kelas serangan secara kumulatif hanya mencakup 5,40%. Ketimpangan ini mencerminkan kondisi realistis lalu lintas jaringan di lingkungan operasional, di mana trafik normal secara alamiah jauh lebih mendominasi dibandingkan trafik serangan.

> **[GAMBAR 4.1 — Diagram Batang/Pie Distribusi Kelas Dataset]**
> *Deskripsi: Visualisasi proporsi keempat kelas (Normal, DoS, Probe, Malware) yang menggambarkan ketidakseimbangan ekstrem pada dataset. File referensi: `distribusi_dan_bobot_skripsi.png`.*

Gambar 4.1 memvisualisasikan distribusi keempat kelas secara grafis, memperjelas dominasi kelas Normal yang menempati lebih dari 94% ruang diagram. Secara kuantitatif, **rasio ketidakseimbangan mencapai 122,28 : 1** antara kelas mayoritas (Normal = 2.237.731 sampel) dan kelas minoritas terkecil (Probe = 18.300 sampel). Rasio di atas 100:1 ini tergolong *extreme imbalance* yang dapat menyebabkan classifier bias terhadap kelas mayoritas jika tidak ditangani melalui strategi khusus seperti pembobotan kelas atau *resampling*.

Kode Program 4.2 menunjukkan proses pemuatan dataset dan pemetaan 10 kategori serangan asli menjadi 4 kelas yang digunakan dalam penelitian ini.

> **[KODE PROGRAM 4.2 — Pemuatan Dataset & Pemetaan Kelas]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 2*

```
 1  df_full = pd.read_csv(data_path)
 2
 3  mapping_rules = {
 4      'Benign': 0,
 5      'DoS': 1, 'Generic': 1,
 6      'Reconnaissance': 2, 'Analysis': 2,
 7      'Exploits': 3, 'Fuzzers': 3, 'Backdoor': 3,
 8      'Shellcode': 3, 'Worms': 3
 9  }
10
11  df_full['mapped_label'] = df_full['Attack'].map(mapping_rules)
12  df_full['mapped_label'] = df_full['mapped_label'].astype(int)
13
14  df_train_full, df_test_raw = train_test_split(
15      df_full, test_size=0.2, random_state=42,
16      stratify=df_full['mapped_label']
17  )
```

Kode Program 4.2 menegaskan alur awal data: dataset dibaca dari sumber utama, label serangan dipetakan menjadi empat kelas target, lalu dilakukan pemisahan train-test secara stratifikasi agar distribusi kelas tetap konsisten pada kedua subset.

---

## 4.2 Hasil Pra-pemrosesan Data

Kualitas data yang masuk ke dalam pipeline pelatihan secara langsung menentukan kualitas model yang dihasilkan. Sub-bab ini menyajikan hasil tahap pra-pemrosesan yang mencakup eliminasi fitur-fitur yang berpotensi menyebabkan bias atau kebocoran data, pembersihan nilai hilang serta nilai tak hingga, dan verifikasi kesiapan seluruh fitur untuk pemodelan. Setiap langkah dilakukan secara sistematis mengikuti prosedur yang telah dirancang pada Bab III, dengan tujuan menghasilkan dataset bersih yang memungkinkan model XGBoost mempelajari pola perilaku lalu lintas jaringan secara objektif.

### 4.2.1 Hasil Pembersihan Fitur dan Data Hilang

Tahap pra-pemrosesan bertujuan menghilangkan informasi yang berpotensi menyebabkan bias, kebocoran data (*data leakage*), atau penurunan kemampuan generalisasi model. Langkah pertama adalah eliminasi fitur-fitur yang tidak relevan untuk tugas klasifikasi intrusi. Sebanyak **5 kolom** dihapus dari dataset dengan justifikasi yang dirinci pada Tabel 4.4.

> **[TABEL 4.4 — Kolom yang Dihapus dari Dataset]**

| No | Nama Kolom | Alasan Penghapusan |
|---|---|---|
| 1 | `FLOW_START_MILLISECONDS` | Metadata temporal yang tidak merepresentasikan karakteristik intrinsik aliran jaringan |
| 2 | `FLOW_END_MILLISECONDS` | Metadata temporal yang tidak merepresentasikan karakteristik intrinsik aliran jaringan |
| 3 | `IPV4_SRC_ADDR` | Identitas IP sumber; berisiko menyebabkan overfitting terhadap alamat spesifik |
| 4 | `IPV4_DST_ADDR` | Identitas IP tujuan; berisiko menyebabkan overfitting terhadap alamat spesifik |
| 5 | `Label` | Label biner redundan; informasi sudah tercakup dalam kolom `Attack` yang telah dipetakan |

Tabel 4.4 mendokumentasikan setiap kolom yang dieliminasi beserta rasional penghapusannya. Penghapusan kolom alamat IP merupakan langkah penting untuk memastikan model mengenali *pola perilaku* lalu lintas — bukan menghafal identitas sumber atau tujuan tertentu — sehingga mampu menggeneralisasi ke lingkungan jaringan yang berbeda dan menghindari *spurious correlation* pada fitur identitas.

Setelah eliminasi kolom, dilakukan pembersihan nilai hilang dan nilai tak hingga. Hasil pembersihan terangkum pada Tabel 4.5.

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

Tabel 4.5 menunjukkan bahwa dari 54 kolom awal, 50 kolom tersisa setelah penghapusan (49 fitur prediktor dan 1 kolom target). Sebanyak 195.882 nilai NaN terdeteksi — yang berasal dari operasi pembagian dengan nol pada beberapa fitur throughput dan durasi — kemudian diimputasi menggunakan **median** dari data training karena sifatnya yang *robust* terhadap *outlier*. Nilai tak hingga (*infinity*) yang dihasilkan dari operasi aritmatika juga berhasil dibersihkan seluruhnya, menghasilkan dataset yang 100% bersih.

Verifikasi otomatis memastikan bahwa seluruh 49 fitur yang tersisa bersifat numerik, sehingga tidak diperlukan proses penyandian kategorikal (*categorical encoding*) tambahan. Daftar lengkap 49 fitur yang digunakan dalam pemodelan disajikan pada Tabel 4.6.

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

Tabel 4.6 memuat 49 fitur yang secara komprehensif merepresentasikan berbagai aspek aliran jaringan. Fitur-fitur ini mencakup: (1) informasi port dan protokol untuk identifikasi layanan, (2) volume lalu lintas masuk dan keluar untuk analisis beban, (3) flag TCP untuk deteksi anomali koneksi, (4) karakteristik temporal aliran untuk profiling durasi, (5) properti paket IP termasuk TTL dan panjang paket, (6) statistik throughput dan retransmisi untuk analisis kualitas koneksi, (7) distribusi ukuran paket untuk identifikasi pola trafik, (8) informasi protokol aplikasi (DNS, FTP, ICMP), serta (9) statistik *Inter-Arrival Time* (IAT) untuk analisis pola temporal paket. Keragaman dimensi fitur ini memungkinkan model menangkap pola serangan dari berbagai sudut pandang perilaku jaringan.

Kode Program 4.3 menampilkan proses pembersihan kolom identitas, penanganan nilai NaN dan infinity, serta validasi kebersihan data.

> **[KODE PROGRAM 4.3 — Pembersihan Kolom & Penanganan NaN/Infinity]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 (bagian pembersihan)*

```
 1  cols_to_drop = [
 2      'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
 3      'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'
 4  ]
 5  df_train_clean = df_train_full.drop(columns=cols_to_drop, errors='ignore')
 6  df_test_clean = df_test_raw.drop(columns=cols_to_drop, errors='ignore')
 7
 8  X_raw = df_train_clean.drop(columns=[target_col], errors='ignore')
 9  y_raw = df_train_clean[target_col]
10
11  X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
12  train_medians = X_raw.median()
13  X_raw.fillna(train_medians, inplace=True)
14  X_test_final_raw.fillna(train_medians, inplace=True)
```

Kode Program 4.3 memperlihatkan tahap sanitasi data yang krusial, yaitu menghapus kolom berisiko bias/leakage, mengonversi nilai tak hingga menjadi NaN, lalu melakukan imputasi median berbasis data latih agar kualitas data tetap terjaga tanpa kebocoran informasi.


---

## 4.3 Hasil Transformasi Data

Setelah data dibersihkan pada tahap pra-pemrosesan, langkah selanjutnya adalah mentransformasikan data ke dalam format yang optimal untuk pelatihan model. Sub-bab ini membahas dua aspek transformasi yang krusial: pertama, strategi pembagian data secara stratifikasi dan standardisasi fitur menggunakan Z-score normalization untuk menjaga integritas statistik antar subset; kedua, penerapan strategi *hybrid cost-sensitive weighting* untuk mengatasi ketidakseimbangan kelas dengan rasio 122,28:1 tanpa mengubah distribusi data asli. Kedua transformasi ini dirancang agar model memperoleh sinyal pelatihan yang seimbang dan mampu mendeteksi berbagai jenis serangan jaringan secara efektif.

### 4.3.1 Hasil Penyandian dan Standardisasi

Agar model mampu menggeneralisasi dengan baik, pembagian data harus mempertahankan proporsi kelas di setiap subset — terutama pada dataset dengan ketidakseimbangan yang ekstrem. Pembagian dilakukan secara **stratifikasi** (*stratified split*) dalam dua tahap, menghasilkan tiga subset yang dirinci pada Tabel 4.7.

> **[TABEL 4.7 — Pembagian Dataset]**

| Subset | Jumlah Sampel | Persentase | Keterangan |
|---|---|---|---|
| Train | 1.513.871 | 64% dari total | 80% dari 80% pertama |
| Validasi | 378.468 | 16% dari total | 20% dari 80% pertama |
| Test (Holdout) | 473.085 | 20% dari total | Tidak tersentuh hingga evaluasi akhir |
| **Total** | **2.365.424** | **100%** | |

Tabel 4.7 mendeskripsikan strategi pembagian dua tahap yang diterapkan. Pada tahap pertama, dataset dibagi 80:20 menjadi gabungan latih-validasi (1.892.339 sampel) dan subset uji (473.085 sampel). Pada tahap kedua, gabungan latih-validasi dibagi kembali 80:20 menjadi data training final (1.513.871) dan validasi (378.468). Pendekatan ini memastikan data test tetap tersegel (*holdout*) sepanjang proses optimasi, sehingga evaluasi akhir benar-benar mengukur kemampuan generalisasi model terhadap data yang belum pernah dilihat.

Verifikasi proporsi kelas pada data uji mengkonfirmasi konsistensi distribusi pasca-stratifikasi:
- Kelas 0 (Normal): 94,60%
- Kelas 1 (DoS): 1,08%
- Kelas 2 (Probe): 0,77%
- Kelas 3 (Malware): 3,54%

Selanjutnya, standardisasi fitur dilakukan menggunakan **StandardScaler** dengan pendekatan *fit-on-train, transform-on-all* untuk mencegah kebocoran informasi statistik dari data validasi maupun test ke dalam proses pelatihan. Ringkasan transformasi tercantum pada Tabel 4.8.

> **[TABEL 4.8 — Ringkasan Transformasi Data]**

| Tahap | Detail |
|---|---|
| Metode Scaling | StandardScaler (Z-score normalization) |
| Fit pada | Data Training (1.513.871 sampel) |
| Transform pada | Train, Validasi, dan Test |
| Konversi tipe data | `float64` → `float32` (hemat memori) |
| Jumlah fitur final | 49 |

Tabel 4.8 mendokumentasikan langkah standardisasi yang diterapkan. StandardScaler mentransformasikan setiap fitur ke distribusi berpusat nol dengan simpangan baku satu ($z = (x - \mu) / \sigma$), di mana parameter $\mu$ dan $\sigma$ dihitung **eksklusif** dari data training. Normalisasi Z-score ini membantu stabilitas numerik dan konvergensi pada proses pelatihan. Konversi tipe data dari `float64` ke `float32` memangkas konsumsi memori hingga 50% tanpa dampak signifikan terhadap presisi klasifikasi.

Kode Program 4.4 menampilkan proses standardisasi fitur dan pembagian data menjadi subset training dan validasi.

> **[KODE PROGRAM 4.4 — Standardisasi & Pembagian Data]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 (bagian standardisasi)*

```
 1  scaler = StandardScaler()
 2  X_enc = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
 3  X_test_enc = pd.DataFrame(scaler.transform(X_test_final_raw),
 4                            columns=X_test_final_raw.columns)
 5
 6  float_cols = X_enc.select_dtypes(include=['float64']).columns
 7  X_enc[float_cols] = X_enc[float_cols].astype('float32')
 8  X_test_enc[float_cols] = X_test_enc[float_cols].astype('float32')
 9
10  X_train, X_val, y_train, y_val = train_test_split(
11      X_selected, y, test_size=0.2, random_state=42, stratify=y
12  )
```

Kode Program 4.4 menunjukkan implementasi standardisasi berbasis Z-score dengan prinsip *fit-on-train, transform-on-all*, dilanjutkan konversi tipe numerik untuk efisiensi memori dan pembagian train-validasi terstratifikasi guna menjaga proporsi kelas.

---

### 4.3.2 Distribusi Bobot Penanganan Imbalance

Ketidakseimbangan kelas dengan rasio 122,28:1 menuntut strategi mitigasi yang tepat agar model tidak bias terhadap kelas mayoritas. Pendekatan *cost-sensitive learning* dipilih karena tidak mengubah distribusi data asli — berbeda dengan metode *resampling* yang berisiko menghilangkan informasi (undersampling) atau memperbanyak noise (oversampling). Strategi **Hybrid Cost-Sensitive Weighting** diterapkan melalui tiga langkah berurutan:

1. **Hitung bobot seimbang (*balanced*)** menggunakan `compute_sample_weight('balanced')` dari scikit-learn, yang memberikan bobot proporsional terbalik terhadap frekuensi kelas.
2. **Transformasi akar kuadrat (`sqrt`)** untuk melunakkan penalti yang terlalu ekstrem pada kelas minoritas, mencegah model menjadi terlalu agresif mengejar kelas langka.
3. **Normalisasi** agar rata-rata bobot = 1,0, menjaga stabilitas *learning rate* XGBoost selama pelatihan.

Hasil distribusi bobot final per kelas ditampilkan pada Tabel 4.9.

> **[TABEL 4.9 — Distribusi Bobot Hybrid per Kelas]**

| Kelas | Jumlah Sampel (Train) | Bobot Balanced (Raw) | Bobot Hybrid (sqrt + norm) | Efektif Sampel |
|---|---|---|---|---|
| Normal | 1.432.148 (94,60%) | 0,2643 | 0,7600 | 1.088.389,79 |
| DoS | 16.404 (1,08%) | 23,0717 | 7,1009 | 116.483,76 |
| Probe | 11.712 (0,77%) | 32,3145 | 8,4038 | 98.425,14 |
| Malware | 53.607 (3,54%) | 7,0600 | 3,9281 | 210.572,31 |

Tabel 4.9 memperlihatkan efek transformasi bobot secara kuantitatif. Tanpa transformasi hybrid, bobot raw untuk kelas Probe (32,3145) terlampau tinggi — setiap sampel Probe bernilai setara dengan 32 sampel Normal, yang dapat menyebabkan *overfitting* pada pola minoritas. Setelah transformasi `sqrt` dan normalisasi, rentang bobot menyempit secara drastis dari 0,26–32,31 menjadi 0,76–8,40, memberikan keseimbangan yang lebih proporsional. Kolom "Efektif Sampel" menunjukkan bagaimana pembobotan mengubah kontribusi relatif setiap kelas: kelas Normal yang awalnya mendominasi 94,60% kini memberikan kontribusi efektif yang lebih proporsional, sementara kelas Probe yang paling langka meningkat kontribusi efektifnya dari ~12 ribu menjadi ~98 ribu.


> **[GAMBAR 4.2 — Dampak Pembobotan terhadap Distribusi Kelas]**
> *Deskripsi: Perbandingan visual distribusi kelas sebelum dan sesudah penerapan bobot hybrid, menggambarkan penurunan dominasi kelas Normal dan peningkatan representasi kelas serangan. File referensi: `impact_weighting_skripsi.png`.*

Gambar 4.2 menyajikan perbandingan *before-after* yang menunjukkan pergeseran distribusi secara keseluruhan. Sebelum pembobotan, kelas Normal secara visual mendominasi hampir seluruh area diagram, yang berarti gradien pelatihan didominasi oleh pola lalu lintas normal. Setelah pembobotan, kontribusi kelas serangan meningkat secara proporsional, memberikan sinyal yang lebih kuat kepada algoritma untuk mempelajari pola-pola anomali yang jarang namun kritis untuk dideteksi.

Statistik ringkasan bobot mengkonfirmasi kewajaran distribusi:
- **Min Weight:** 0,7600 (Normal — kelas mayoritas, mendapat penalti rendah)
- **Max Weight:** 8,4038 (Probe — kelas paling minoritas, mendapat penalti tertinggi)
- **Mean Weight:** 1,0000 (ternormalisasi, menjaga stabilitas gradien)

Kode Program 4.5 menampilkan implementasi strategi *Hybrid Cost-Sensitive Weighting* yang terdiri dari tiga langkah berurutan.

> **[KODE PROGRAM 4.5 — Perhitungan Bobot Hybrid Cost-Sensitive]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 (bagian pembobotan)*

```
1  raw_weights = compute_sample_weight(class_weight='balanced', y=y_train)
2  sample_weights_train = np.sqrt(raw_weights)
3  sample_weights_train = sample_weights_train / sample_weights_train.mean()
```

Kode Program 4.5 merangkum strategi pembobotan hybrid: bobot seimbang dihitung dari distribusi kelas, dilunakkan dengan akar kuadrat agar tidak terlalu ekstrem, lalu dinormalisasi sehingga rata-rata bobot tetap 1 dan proses pelatihan lebih stabil.


---

## 4.4 Hasil Optimasi Model Multi-Objective

Optimasi hiperparameter merupakan inti dari penelitian ini, di mana tiga metode sampling — TPE, NSGA-II, dan Random Search — dibandingkan kemampuannya dalam menavigasi ruang pencarian berdimensi tinggi untuk menemukan konfigurasi XGBoost yang secara simultan memaksimalkan *Macro F1-Score* dan meminimalkan *Inference Latency*. Sub-bab ini menyajikan hasil eksperimen optimasi multi-objective secara menyeluruh, mulai dari dinamika pencarian selama 90 trial (30 trial per metode), pola konvergensi yang membedakan karakteristik tiap algoritma, hingga analisis komparatif terhadap konfigurasi hiperparameter optimal yang ditemukan. Pendekatan multi-objective ini dilandasi oleh teori optimasi Pareto yang mengakui bahwa pada permasalahan dengan objektif berkonflik, tidak ada satu solusi tunggal yang optimal — melainkan sekumpulan solusi *trade-off* yang masing-masing merepresentasikan kompromi terbaik antara kualitas deteksi dan efisiensi komputasi.

### 4.4.1 Dinamika Pencarian Hiperparameter

Performa model XGBoost sangat bergantung pada konfigurasi hiperparameter. Berbeda dengan pendekatan optimasi hiperparameter konvensional yang hanya mengoptimalkan satu metrik, penelitian ini menerapkan pendekatan **multi-objective** yang mengoptimalkan dua fungsi tujuan secara simultan melalui framework Optuna:

- **Objective 1:** Maksimasi *Macro F1-Score* — mengukur kualitas deteksi secara seimbang antar kelas.
- **Objective 2:** Minimasi *Inference Latency* (µs/sample) — mengukur kecepatan prediksi yang krusial untuk penerapan *real-time*.

Tiga metode sampling Optuna dibandingkan untuk mengeksplorasi ruang pencarian, masing-masing menjalankan 30 trial. Perbandingan ketiga metode disajikan pada Tabel 4.10.

> **[TABEL 4.10 — Perbandingan Metode Sampling Optuna]**

| Metode | Tipe Algoritma | Konfigurasi | Jumlah Trial | Waktu Optimasi (detik) | Waktu Optimasi (menit) | Solusi Pareto |
|---|---|---|---|---|---|---|
| TPE | Bayesian (Tree-structured Parzen Estimator) | `TPESampler(seed=42)` | 30 | 1.488,16 | 24,80 | 5 |
| NSGA-II | Evolutionary Algorithm | `NSGAIISampler(seed=42, population_size=10)` | 30 | 1.883,34 | 31,39 | 4 |
| Random | Random Search (Baseline) | `RandomSampler(seed=42)` | 30 | 2.041,33 | 34,02 | 6 |

Tabel 4.10 memperlihatkan bahwa TPE menyelesaikan 30 trial paling cepat (24,80 menit), diikuti NSGA-II (31,39 menit) dan Random (34,02 menit). Efisiensi TPE disebabkan oleh mekanisme *surrogate model*-nya yang memfokuskan sampling pada region yang menjanjikan. Menariknya, Random Search justru menemukan jumlah solusi Pareto terbanyak (6 solusi), mengindikasikan bahwa eksplorasi acak yang luas mampu menjangkau lebih banyak titik *trade-off* yang beragam.

Secara keseluruhan, **90 model** dilatih selama proses optimasi (~90 menit total waktu komputasi GPU). Ruang pencarian hiperparameter yang didefinisikan untuk ketiga metode bersifat identik, memastikan perbandingan yang adil. Detail ruang pencarian disajikan pada Tabel 4.11.

> **[TABEL 4.11 — Ruang Pencarian Hiperparameter]**

| Parameter | Tipe | Rentang | Fungsi |
|---|---|---|---|
| `n_estimators` | Integer | [500, 2000], step=100 | Jumlah pohon; menentukan kompleksitas dan kapasitas model |
| `learning_rate` | Float (log) | [0,01 – 0,3] | Laju pembelajaran; mengatur kecepatan konvergensi per iterasi |
| `max_depth` | Integer | [6, 12] | Kedalaman maksimum pohon; mengontrol kapasitas penangkapan pola |
| `min_child_weight` | Integer | [1, 7] | Bobot minimum pada node daun; regularisasi terhadap split minor |
| `max_delta_step` | Integer | [1, 8] | Batas perubahan output; menstabilkan gradien pada kelas tak seimbang |
| `gamma` | Float | [0,1 – 0,5] | Minimum *loss reduction* untuk split; regularisasi terhadap pohon kompleks |
| `subsample` | Float | [0,6 – 0,95] | Fraksi sampel per pohon; teknik *bagging* untuk reduksi varians |
| `colsample_bytree` | Float | [0,5 – 0,9] | Fraksi fitur per pohon; diversifikasi antar pohon |
| `reg_alpha` (L1) | Float (log) | [1e-6 – 1,0] | Regularisasi L1; mendorong sparsity pada bobot fitur |
| `reg_lambda` (L2) | Float (log) | [1e-6 – 1,0] | Regularisasi L2; menghaluskan bobot untuk mencegah overfitting |

Tabel 4.11 mendefinisikan 10 hiperparameter yang dioptimasi beserta rasional pemilihan rentangnya. Parameter `learning_rate` dan regularisasi (`reg_alpha`, `reg_lambda`) menggunakan skala logaritmik, sementara `max_delta_step` dimasukkan untuk menstabilkan pembaruan gradien pada kasus ketidakseimbangan kelas ekstrem. Rentang `n_estimators` yang lebar ([500, 2000]) dirancang untuk memberikan ruang eksplorasi *trade-off* antara akurasi dan kecepatan inferensi.

Kode Program 4.6 menampilkan definisi fungsi objective multi-objective yang digunakan oleh Optuna, mencakup sampling hiperparameter, pelatihan model, dan pengukuran kedua objektif.

> **[KODE PROGRAM 4.6 — Fungsi Objective Multi-Objective]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 4*

```
 1  def objective_xgboost_multi(trial):
 2      param = {
 3          'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
 4          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
 5          'max_depth': trial.suggest_int('max_depth', 6, 12),
 6          'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
 7          'max_delta_step': trial.suggest_int('max_delta_step', 1, 8),
 8          'gamma': trial.suggest_float('gamma', 0.1, 0.5),
 9          'subsample': trial.suggest_float('subsample', 0.6, 0.95),
10          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
11          'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
12          'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
13          'objective': 'multi:softmax', 'num_class': NUM_CLASSES,
14          'tree_method': 'hist', 'device': 'cuda',
15          'eval_metric': 'mlogloss', 'verbosity': 0, 'random_state': 42
16      }
17
18      model = xgb.XGBClassifier(**param)
19      model.fit(X_train, y_train, sample_weight=sample_weights_train,
20                eval_set=[(X_val, y_val)], verbose=False)
21
22      start_time = time.perf_counter()
23      preds = model.predict(X_val)
24      end_time = time.perf_counter()
25      latency_us = ((end_time - start_time) / len(X_val)) * 1_000_000
26
27      f1_macro = f1_score(y_val, preds, average='macro')
28
29      return f1_macro, latency_us
```

Kode Program 4.6 menggambarkan fungsi objektif multi-objective yang menjadi inti optimasi: setiap trial menghasilkan kombinasi hiperparameter baru, model dilatih pada data berbobot, lalu dievaluasi berdasarkan dua sasaran sekaligus, yaitu F1 Macro (maksimasi) dan latensi inferensi (minimasi).

Kode Program 4.7 menampilkan konfigurasi dan eksekusi optimasi menggunakan tiga metode sampling Optuna secara berurutan.

> **[KODE PROGRAM 4.7 — Konfigurasi & Eksekusi Optimasi Optuna]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 5.0 dan Cell 5.2*

```
 1  N_TRIALS = 30
 2  SEED = 42
 3
 4  study_tpe = optuna.create_study(
 5      study_name="TPE_MultiObjective",
 6      directions=["maximize", "minimize"],
 7      sampler=TPESampler(seed=SEED)
 8  )
 9  study_tpe.optimize(objective_xgboost_multi, n_trials=N_TRIALS)
10
11  study_nsga = optuna.create_study(
12      study_name="NSGA2_MultiObjective",
13      directions=["maximize", "minimize"],
14      sampler=NSGAIISampler(seed=SEED, population_size=10)
15  )
16  study_nsga.optimize(objective_xgboost_multi, n_trials=N_TRIALS)
17
18  study_random = optuna.create_study(
19      study_name="Random_MultiObjective",
20      directions=["maximize", "minimize"],
21      sampler=RandomSampler(seed=SEED)
22  )
23  study_random.optimize(objective_xgboost_multi, n_trials=N_TRIALS)
```

Kode Program 4.7 menegaskan desain eksperimen yang adil, di mana ketiga sampler Optuna (TPE, NSGA-II, dan Random) dijalankan pada jumlah trial dan seed yang sama sehingga perbandingan performa antar metode dilakukan dalam kondisi yang setara.

> **[GAMBAR 4.3 — Grafik Riwayat Optimasi (Optimization History)]**
> *Deskripsi: Grafik garis yang menunjukkan dinamika pencarian per trial untuk ketiga metode, menampilkan evolusi F1-Score dan Latency sepanjang 30 trial. File referensi: `optimization_history_final.png`.*

Gambar 4.3 memvisualisasikan trajektori pencarian dari seluruh 90 trial pada dua sumbu objektif. Pola konvergensi yang berbeda terlihat jelas: TPE menunjukkan perbaikan bertahap yang relatif konsisten seiring berjalannya trial, mencerminkan kemampuannya memanfaatkan informasi trial sebelumnya (*exploitation*). NSGA-II memperlihatkan fluktuasi yang lebih besar akibat mekanisme mutasi dan *crossover* yang melekat pada algoritma evolusioner. Random Search menampilkan sebaran paling acak, namun sesekali menemukan konfigurasi yang sangat kompetitif — sebuah fenomena yang mendemonstrasikan kekuatan eksplorasi stokastik murni.

Efektivitas pencarian ketiga metode dirangkum melalui statistik konvergensi pada Tabel 4.12.

> **[TABEL 4.12 — Statistik Konvergensi Optimasi]**

| Metode | Gain F1 (%) | Gain Latency (%) | Best F1 | Best Latency (µs) |
|---|---|---|---|---|
| TPE | 1,67 | 50,01 | 0,86 | 3,70 |
| NSGA-II | 1,47 | 47,63 | 0,86 | 3,88 |
| Random | 1,82 | 47,57 | 0,87 | 3,89 |

Tabel 4.12 mengkuantifikasi peningkatan performa dari trial awal ke trial terbaik. Ketiga metode berhasil meningkatkan F1-Score sebesar 1,47–1,82% dan mereduksi latensi sebesar 47,57–50,01%. TPE mencapai reduksi latensi terbesar (50,01%) berkat kemampuannya mengarahkan pencarian ke region model ringan (jumlah pohon sedikit). Random Search meraih peningkatan F1 tertinggi (1,82%), menunjukkan bahwa diversitas eksplorasi yang tinggi sesekali menghasilkan konfigurasi yang sulit ditemukan oleh metode terarah.

---

### 4.4.2 Konfigurasi Hiperparameter Optimal

Konfigurasi hiperparameter dari solusi Pareto terbaik (F1 tertinggi) untuk setiap metode dibandingkan secara langsung pada Tabel 4.13, memungkinkan analisis strategi yang berbeda antar algoritma dalam menavigasi ruang pencarian.

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

Tabel 4.13 mengungkap beberapa pola menarik yang menghubungkan konfigurasi hiperparameter dengan perilaku model:

1. **Learning rate rendah secara konsisten** (0,0147–0,0235) di semua metode mengindikasikan bahwa konvergensi lambat menghasilkan generalisasi yang lebih baik pada dataset ini dan mengurangi risiko overfitting.

2. **Trade-off jumlah pohon dan latensi terlihat jelas**: TPE menemukan model paling ringan (600 pohon) yang menjelaskan keunggulan latensinya (2,05 µs), sementara NSGA-II menggunakan model terberat (1.400 pohon) dengan latensi tertinggi (3,84 µs). Hubungan linear ini konsisten dengan kompleksitas inferensi $O(K \cdot D)$ di mana $K$ adalah jumlah pohon dan $D$ kedalaman.

3. **Strategi regularisasi yang berbeda**: TPE mengandalkan regularisasi L1 agresif (`reg_alpha=0,8388`) yang mendorong *sparsity* pada fitur, sementara NSGA-II dan Random menggunakan regularisasi minimal — mengisyaratkan bahwa terdapat beberapa jalur regulasi berbeda menuju performa serupa.

4. **Gamma moderat** (0,35–0,41) pada semua metode menunjukkan kebutuhan *pruning* yang konsisten, memastikan setiap split pada pohon memberikan kontribusi minimum terhadap penurunan *loss*.


---

## 4.5 Evaluasi Kinerja Model

Evaluasi kinerja model merupakan tahap kritis yang menentukan validitas dan reliabilitas seluruh proses optimasi yang telah dilakukan. Sub-bab ini menyajikan evaluasi komprehensif melalui tiga perspektif analitis yang saling melengkapi: analisis Pareto front untuk mengidentifikasi trade-off optimal antara kualitas deteksi dan kecepatan inferensi, analisis kesalahan klasifikasi melalui confusion matrix dan Cohen's Kappa untuk mengukur reliabilitas prediksi, serta validasi signifikansi statistik menggunakan uji Kruskal-Wallis pada hasil cross-validation untuk memastikan bahwa perbedaan performa yang teramati bersifat genuine — bukan sekadar fluktuasi stokastik.

### 4.5.1 Analisis Pareto Front (Trade-off Solusi)

Suatu solusi dikatakan *Pareto-optimal* atau *non-dominated* jika tidak ada solusi lain yang lebih baik pada semua objektif secara bersamaan. Himpunan seluruh solusi non-dominated membentuk *Pareto front*, yang merepresentasikan *trade-off* terbaik yang dapat dicapai antara objektif yang berkonflik.

> **[GAMBAR 4.4 — Pareto Front Statis (Ketiga Metode)]**
> *Deskripsi: Scatter plot yang menampilkan seluruh trial (titik transparan) dan solusi Pareto-optimal (titik tebal berwarna) untuk TPE (biru), NSGA-II (merah), dan Random (hijau) pada ruang F1-Score vs Latency. File referensi: `pareto_front_static_hd.png`.*

Gambar 4.4 merupakan visualisasi kunci yang menempatkan seluruh 90 trial dan 15 solusi Pareto-optimal pada bidang koordinat F1-Score × Latensi. Beberapa temuan empiris yang terlihat dari visualisasi ini: (1) *Pareto front* ketiga metode berada pada region yang berdekatan, mengindikasikan bahwa tidak ada satu metode yang secara konsisten mendominasi metode lain pada kedua objektif; (2) TPE menguasai region latensi rendah (sisi kiri grafik) dengan solusi-solusi yang efisien secara komputasi; (3) Random Search menjangkau titik F1 tertinggi (sisi atas grafik) meskipun dengan latensi lebih besar; (4) NSGA-II menghasilkan *front* yang paling kompak dengan variasi antar solusi yang lebih terbatas.

Detail solusi Pareto-optimal untuk setiap metode disajikan pada Tabel 4.14 hingga Tabel 4.16.

> **[TABEL 4.14 — Solusi Pareto-Optimal TPE (5 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 26 | 0,8648 | 2,05 | 600 | 0,0235 | 10 |
| 2 | 21 | 0,8633 | 1,72 | 700 | 0,0652 | 7 |
| 3 | 22 | 0,8615 | 1,62 | 700 | 0,0748 | 6 |
| 4 | 4 | 0,8594 | 1,46 | 600 | 0,0539 | 6 |
| 5 | 23 | 0,8565 | 1,40 | 500 | 0,0266 | 7 |

Tabel 4.14 memuat 5 solusi Pareto-optimal yang ditemukan TPE, membentuk *trade-off* yang terstruktur: solusi #1 memberikan F1 tertinggi (0,8648) namun dengan latensi terbesar (2,05 µs), sedangkan solusi #5 menawarkan latensi terendah (1,40 µs) dengan kompromi pada F1 (0,8565). Keseluruhan rentang latensi TPE (1,40–2,05 µs) secara konsisten lebih rendah dibandingkan metode lain, yang disebabkan oleh preferensi TPE terhadap model dengan jumlah pohon sedikit (500–700) — sebuah artefak dari mekanisme *exploitation* Bayesian yang cenderung memperdalam pencarian di sekitar konfigurasi ringan yang terbukti efisien.

> **[TABEL 4.15 — Solusi Pareto-Optimal NSGA-II (4 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 2 | 0,8631 | 3,84 | 1.400 | 0,0161 | 8 |
| 2 | 7 | 0,8601 | 3,14 | 1.700 | 0,0197 | 6 |
| 3 | 20 | 0,8597 | 3,12 | 1.700 | 0,0197 | 6 |
| 4 | 4 | 0,8594 | 1,47 | 600 | 0,0539 | 6 |

Tabel 4.15 menunjukkan 4 solusi Pareto NSGA-II dengan karakteristik yang berbeda dari TPE. NSGA-II cenderung menghasilkan model yang lebih berat — tiga dari empat solusinya menggunakan 1.400–1.700 pohon, menghasilkan latensi 3,12–3,84 µs. Hanya solusi #4 (yang kebetulan merupakan solusi identik dengan TPE Trial #4) yang menggunakan konfigurasi ringan. Pola ini mencerminkan mekanisme evolusioner NSGA-II yang mempertahankan populasi beragam namun, dengan `population_size=10` yang relatif kecil, konvergen ke region model berat yang memberikan jaminan akurasi lebih stabil.

> **[TABEL 4.16 — Solusi Pareto-Optimal Random (6 Solusi)]**

| # | Trial ID | Macro F1 | Latensi (µs) | n_estimators | learning_rate | max_depth |
|---|---|---|---|---|---|---|
| 1 | 18 | 0,8660 | 4,00 | 1.000 | 0,0147 | 12 |
| 2 | 11 | 0,8655 | 3,64 | 900 | 0,0173 | 12 |
| 3 | 25 | 0,8641 | 3,24 | 900 | 0,0371 | 11 |
| 4 | 29 | 0,8627 | 2,02 | 500 | 0,0114 | 11 |
| 5 | 10 | 0,8621 | 1,50 | 500 | 0,0871 | 8 |
| 6 | 4 | 0,8594 | 1,47 | 600 | 0,0539 | 6 |

Tabel 4.16 mengungkap bahwa Random Search berhasil menemukan jumlah solusi Pareto terbanyak (6 solusi) dengan rentang F1 terlebar (0,8594–0,8660). Temuan menonjol adalah bahwa dua solusi teratas (Trial #18 dan #11) keduanya menggunakan `max_depth=12` — kedalaman maksimum dalam ruang pencarian — yang mengisyaratkan bahwa pohon yang lebih dalam diperlukan untuk menangkap interaksi fitur yang kompleks pada pola serangan tertentu. Solusi #1 (F1=0,8660) merupakan skor validasi F1 tertinggi yang ditemukan di antara seluruh 90 trial dari ketiga metode.

Model Pareto terbaik dari masing-masing metode kemudian dilatih ulang (*retrain*) menggunakan seluruh data training dan dievaluasi pada data test *holdout*. Hasil evaluasi akhir dirangkum pada Tabel 4.17.

> **[TABEL 4.17 — Perbandingan Kinerja Akhir pada Data Test]**

| Metrik | NSGA-II | TPE | Random |
|---|---|---|---|
| **F1 Macro** | 0,8614 | 0,8629 | **0,8642** |
| **Accuracy** | 0,9925 | 0,9926 | **0,9927** |
| **Latency (µs/sample)** | 3,98 | **2,23** | 4,15 |
| **Training Time (s)** | 70,05 | **34,70** | 66,19 |

Tabel 4.17 merangkum empat metrik kinerja yang menentukan. Dari perspektif kualitas deteksi, Random unggul tipis pada F1 Macro (0,8642) dan Accuracy (99,27%), namun selisih antar metode sangat kecil (rentang F1: 0,0028 atau 0,28%). Dari perspektif efisiensi operasional, TPE mendominasi dengan keunggulan yang signifikan: latensi inferensi hampir dua kali lipat lebih cepat (2,23 vs 4,15 µs) dan waktu pelatihan kurang dari setengah (34,70 vs 66,19 detik). Disparitas ini merupakan konsekuensi langsung dari jumlah pohon yang digunakan — 600 pohon (TPE) versus 1.000 (Random) dan 1.400 (NSGA-II).

Kode Program 4.8 menampilkan proses retrain model dengan parameter terbaik dari Pareto front dan evaluasi pada data test holdout.

> **[KODE PROGRAM 4.8 — Evaluasi Final & Retrain Model Terbaik]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 8*

```
 1  def train_and_evaluate(study, name):
 2      pareto_front = study.best_trials
 3      best_trial = max(pareto_front, key=lambda t: t.values[0])
 4      best_params = best_trial.params
 5
 6      best_params.update({
 7          'objective': 'multi:softmax', 'num_class': len(target_names),
 8          'tree_method': 'hist', 'device': 'cuda',
 9          'eval_metric': 'mlogloss', 'verbosity': 0, 'random_state': 42
10      })
11
12      model = xgb.XGBClassifier(**best_params)
13      model.fit(X_train, y_train, sample_weight=sample_weights_train,
14                eval_set=[(X_val, y_val)], verbose=False)
15
16      _ = model.predict(X_test_selected.iloc[:100])
17      start_inf = time.perf_counter()
18      preds = model.predict(X_test_selected)
19      end_inf = time.perf_counter()
20      latency_us = ((end_inf - start_inf) / len(X_test_selected)) * 1_000_000
21
22      return model, preds, latency_us
```

Kode Program 4.8 menunjukkan prosedur evaluasi final yang sistematis: memilih trial Pareto terbaik berdasarkan F1, melatih ulang model dengan parameter terpilih, lalu menghitung prediksi dan latensi pada data uji holdout untuk menghasilkan metrik komparatif akhir.

> **[GAMBAR 4.5 — Diagram Batang Perbandingan Metrik antar Metode]**
> *Deskripsi: Grouped bar chart yang membandingkan F1-Macro, Accuracy, Latency, dan Training Time untuk ketiga metode secara berdampingan. File referensi: `metrics_grouped_bar.png`.*

Gambar 4.5 memvisualisasikan keempat metrik dalam format *grouped bar chart* yang memungkinkan perbandingan langsung antar metode. Representasi visual ini memperjelas temuan kuantitatif dari Tabel 4.17: batang F1 dan Accuracy memiliki ketinggian yang hampir tidak dapat dibedakan secara visual, sementara batang Latency dan Training Time menunjukkan perbedaan yang mencolok — mengkonfirmasi bahwa diferensiasi utama antar metode terletak pada **efisiensi komputasi**, bukan kualitas deteksi.

> **[GAMBAR 4.6 — Heatmap F1-Score per Kelas per Metode]**
> *Deskripsi: Heatmap warna yang menampilkan F1-Score setiap kelas (Normal, DoS, Probe, Malware) untuk ketiga metode optimasi, memungkinkan identifikasi kelas yang paling sulit diklasifikasikan. File referensi: `metrics_f1_heatmap.png`.*

Gambar 4.6 menyajikan dekomposisi F1-Score ke level per-kelas dalam format heatmap. Gradasi warna mengungkap hierarki kesulitan klasifikasi yang konsisten di seluruh metode: Normal mencapai F1 sempurna (1,0000, warna terdalam), Malware berada di tingkat tinggi (~0,90), DoS di tingkat menengah (~0,82), dan Probe secara konsisten menjadi kelas tersulit (~0,73, warna terpudar). Keseragaman pola ini lintas metode menunjukkan bahwa tingkat kesulitan bersifat **inheren terhadap karakteristik data**, bukan merupakan keterbatasan metode optimasi tertentu.

> **[GAMBAR 4.7 — Scatter Plot Precision vs Recall per Kelas]**
> *Deskripsi: Scatter plot yang memposisikan setiap kombinasi kelas-metode pada ruang Precision × Recall untuk menganalisis trade-off deteksi. File referensi: `metrics_pr_scatter.png`.*

Gambar 4.7 menempatkan setiap kombinasi kelas-metode pada ruang dua dimensi Precision × Recall. Titik-titik yang mendekati pojok kanan atas merepresentasikan kinerja ideal. Pola spasial yang teramati mengungkap karakteristik deteksi yang berbeda per kelas: Normal berada tepat di titik ideal (1,0; 1,0); Malware menunjukkan recall lebih tinggi dari precision, mengindikasikan sensitivitas model yang tinggi terhadap kelas ini dengan konsekuensi sedikit *false positive*; DoS memperlihatkan precision lebih tinggi dari recall, menandakan model bersikap konservatif — lebih memilih menghindari alarm palsu meskipun beberapa serangan DoS lolos; Probe berada di posisi paling jauh dari titik ideal, mengkonfirmasi statusnya sebagai kelas tersulit.

#### Laporan Klasifikasi Detail per Metode

Untuk memberikan gambaran lengkap kinerja per-kelas, laporan klasifikasi detail dari setiap metode disajikan pada Tabel 4.18 hingga 4.20.

> **[TABEL 4.18 — Classification Report NSGA-II (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8831 | 0,7563 | 0,8148 | 5.126 |
| Probe | 0,7301 | 0,7257 | 0,7279 | 3.660 |
| Malware | 0,8834 | 0,9235 | 0,9030 | 16.753 |
| **Macro avg** | **0,8742** | **0,8514** | **0,8614** | 473.085 |
| Weighted avg | 0,9925 | 0,9925 | 0,9925 | 473.085 |

Tabel 4.18 merinci kinerja model NSGA-II pada data test. Kelas Normal mencapai skor sempurna pada semua metrik (precision = recall = F1 = 1,0000), artinya model tidak pernah salah mengklasifikasikan trafik normal sebagai serangan maupun melewatkan trafik normal. Kelas Malware menunjukkan kinerja kuat (F1=0,9030) dengan recall (0,9235) yang lebih tinggi dari precision (0,8834) — sebuah profil yang diinginkan untuk NIDS karena mendeteksi sebanyak mungkin malware lebih penting daripada menghindari beberapa alarm palsu. Kelas DoS memiliki F1=0,8148 dengan kesenjangan recall yang lebih besar (0,7563), mengindikasikan bahwa sekitar 24% serangan DoS tidak terdeteksi. Kelas Probe menjadi yang terlemah (F1=0,7279) akibat kemiripan pola fitur dengan kelas lain.

> **[TABEL 4.19 — Classification Report TPE (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8883 | 0,7583 | 0,8181 | 5.126 |
| Probe | 0,7301 | 0,7281 | 0,7291 | 3.660 |
| Malware | 0,8846 | 0,9249 | 0,9043 | 16.753 |
| **Macro avg** | **0,8758** | **0,8528** | **0,8629** | 473.085 |
| Weighted avg | 0,9926 | 0,9926 | 0,9925 | 473.085 |

Tabel 4.19 menunjukkan model TPE dengan pola serupa namun peningkatan marginal di semua kelas serangan: DoS F1 naik menjadi 0,8181 (+0,0033 dari NSGA-II), Probe F1 naik menjadi 0,7291 (+0,0012), dan Malware F1 naik ke 0,9043 (+0,0013). Meskipun peningkatannya kecil secara absolut, konsistensi perbaikan di setiap kelas menunjukkan bahwa konfigurasi hiperparameter TPE (600 pohon, `max_depth=10`) memberikan keseimbangan kapasitas model yang sedikit lebih optimal.

> **[TABEL 4.20 — Classification Report Random (Test Set)]**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 1,0000 | 1,0000 | 1,0000 | 447.546 |
| DoS | 0,8907 | 0,7628 | 0,8218 | 5.126 |
| Probe | 0,7452 | 0,7137 | 0,7291 | 3.660 |
| Malware | 0,8829 | 0,9299 | 0,9058 | 16.753 |
| **Macro avg** | **0,8797** | **0,8516** | **0,8642** | 473.085 |
| Weighted avg | 0,9927 | 0,9927 | 0,9926 | 473.085 |

Tabel 4.20 menunjukkan model Random sebagai pemegang skor F1 Macro tertinggi (0,8642). Profil per-kelasnya memperlihatkan nuansa yang menarik: precision DoS tertinggi di antara ketiga metode (0,8907), namun precision tersebut dicapai dengan pengorbanan recall Probe yang justru terendah (0,7137). Kelas Malware meraih recall tertinggi (0,9299), mengisyaratkan bahwa konfigurasi `max_depth=12` pada model Random mampu menangkap pola-pola eksploitasi yang lebih kompleks. Perbedaan antar metode pada level per-kelas tetap berada pada orde 0,001–0,003, memperkuat argumentasi bahwa ketiganya secara substansial setara dalam kemampuan deteksi.

> **[GAMBAR 4.8 — Ringkasan Recall per Kelas untuk Ketiga Metode]**
> *Deskripsi: Bar chart horizontal yang membandingkan recall (detection rate) setiap kelas serangan antar metode, memfokuskan pada kemampuan deteksi aktual. File referensi: `recall_summary_chart.png`.*

Gambar 4.8 memfokuskan perbandingan pada metrik **recall** — yang dalam konteks NIDS merepresentasikan *detection rate* atau proporsi serangan yang berhasil diidentifikasi. Visualisasi ini mengungkap bahwa Malware memiliki detection rate tertinggi (>92%) di semua metode, DoS berada di tingkat menengah (~76%), dan Probe memiliki detection rate terendah (~71–73%). Rendahnya recall Probe berkaitan erat dengan fenomena *class overlap* di mana pola *reconnaissance* (scanning, probing) sering kali menyerupai perilaku eksploitasi.


---

### 4.5.2 Analisis Kesalahan dan Reliabilitas (Confusion Matrix & Kappa)

#### Confusion Matrix

Confusion matrix memetakan distribusi prediksi model terhadap label sebenarnya untuk mengidentifikasi pola kesalahan spesifik yang tidak tertangkap oleh metrik agregat seperti F1-Score.

> **[GAMBAR 4.9 — Confusion Matrix Raw (Ketiga Metode)]**
> *Deskripsi: Tiga heatmap confusion matrix berdampingan (NSGA-II, TPE, Random) yang menampilkan jumlah absolut prediksi per sel. Diagonal utama menunjukkan prediksi benar, sel off-diagonal menunjukkan kesalahan klasifikasi. File referensi: `cm_raw_heatmap.png`.*

Gambar 4.9 menampilkan confusion matrix dalam format absolut (jumlah sampel) untuk ketiga metode secara berdampingan. Diagonal utama yang berwarna gelap menandakan dominasi prediksi benar, dengan kelas Normal yang mendominasi secara visual karena volumenya yang sangat besar (447.546 sampel). Sel-sel off-diagonal yang paling menonjol terkonsentrasi pada interaksi antar kelas serangan (DoS-Probe-Malware), sedangkan baris dan kolom Normal hampir seluruhnya bersih — mengkonfirmasi bahwa tantangan klasifikasi berpusat pada pembedaan tipe serangan, bukan pada pemisahan normal versus serangan.

> **[GAMBAR 4.10 — Confusion Matrix Normalized (Ketiga Metode)]**
> *Deskripsi: Tiga heatmap confusion matrix ternormalisasi per baris (recall-based) yang menampilkan proporsi prediksi untuk setiap kelas aktual, menghilangkan efek perbedaan jumlah sampel. File referensi: `cm_norm_heatmap.png`.*

Gambar 4.10 menyajikan versi ternormalisasi yang menghilangkan bias ukuran sampel dan memungkinkan perbandingan proporsional lintas kelas. Normalisasi per baris memastikan bahwa setiap kelas — terlepas dari jumlah sampelnya — dievaluasi pada skala yang sama (0–100%). Dari heatmap ini, pola kesalahan sistematis teridentifikasi secara visual: sel Probe→Malware dan DoS→Malware muncul sebagai area off-diagonal berwarna paling intens, mengkonfirmasi keberadaan *class overlap* yang signifikan.

Kuantifikasi pola kesalahan disajikan pada Tabel 4.21.

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

Tabel 4.21 mengidentifikasi tiga pola kesalahan yang dominan dan konsisten di ketiga metode. Kesalahan terbesar adalah **Probe → Malware** (25,8–27,3%), di mana lebih dari seperempat sampel Probe salah diklasifikasikan sebagai Malware akibat tumpang tindih fitur yang substansial antara aktivitas *reconnaissance* dan eksploitasi. Kesalahan kedua, **DoS → Malware** (20,8–21,2%), disebabkan oleh beberapa varian serangan DoS modern yang mengandung komponen eksploitasi. Kesalahan **Malware → Probe** (4,4–4,9%) memiliki tingkat yang jauh lebih rendah, menunjukkan asimetri: model lebih sering salah mengenali serangan ringan sebagai berat, daripada sebaliknya.

Konsistensi pola kesalahan lintas metode merupakan temuan penting — ini mengindikasikan bahwa sumber kesalahan berasal dari **kemiripan inheren antar kelas dalam ruang fitur**, bukan dari kelemahan algoritma optimasi tertentu. Implikasi praktisnya, peningkatan akurasi pada kelas Probe kemungkinan memerlukan pendekatan di level fitur (misalnya *feature engineering* tambahan) atau arsitektur model, bukan sekadar penyesuaian hiperparameter.

#### Cohen's Kappa

Metrik Cohen's Kappa ($\kappa$) mengukur tingkat kesepakatan antara prediksi model dan label aktual dengan memperhitungkan kesepakatan yang terjadi secara kebetulan, sehingga memberikan evaluasi yang lebih konservatif dibandingkan accuracy pada dataset tidak seimbang. Hasil pengukuran Kappa ditampilkan pada Tabel 4.22.

> **[TABEL 4.22 — Skor Cohen's Kappa per Metode]**

| Metode | Kappa Score | Interpretasi |
|---|---|---|
| Random | 0,9298 | Almost Perfect (Sangat Andal) |
| TPE | 0,9287 | Almost Perfect (Sangat Andal) |
| NSGA-II | 0,9278 | Almost Perfect (Sangat Andal) |

Tabel 4.22 menunjukkan bahwa ketiga metode menghasilkan $\kappa > 0,92$, yang tergolong dalam kategori **"Almost Perfect Agreement"** (rentang 0,81–1,00). Selisih $\kappa$ antar metode sangat kecil (0,0020), mengkonfirmasi kesetaraan reliabilitas klasifikasi.

> **[GAMBAR 4.11 — Diagram Batang Perbandingan Cohen's Kappa]**
> *Deskripsi: Bar chart yang memvisualisasikan skor Kappa ketiga metode dengan garis threshold interpretasi (0,81–1,00 = Almost Perfect). File referensi: `kappa_comparison.png`.*

Gambar 4.11 memvisualisasikan perbandingan $\kappa$ dalam format diagram batang yang dilengkapi garis referensi kategori interpretasi. Ketiga batang memiliki tinggi yang nyaris identik dan seluruhnya berada jauh di atas batas "Almost Perfect" (0,81), memberikan konfirmasi visual yang meyakinkan bahwa seluruh model memiliki tingkat keandalan klasifikasi yang sangat tinggi.

Kode Program 4.9 menampilkan proses perhitungan Cohen's Kappa dan analisis pola kesalahan klasifikasi dari confusion matrix.

> **[KODE PROGRAM 4.9 — Cohen's Kappa & Analisis Pola Kesalahan]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 10B*

```
 1  for name, model in trained_models.items():
 2      preds = model.predict(X_test_selected)
 3      score = cohen_kappa_score(y_test_final, preds)
 4      kappa_data.append({'Metode': name, 'Kappa Score': score})
 5
 6  for name in trained_models.keys():
 7      model = trained_models[name]
 8      preds = model.predict(X_test_selected)
 9      cm = confusion_matrix(y_test_final, preds)
10      errors = []
11      for r in range(len(class_names)):
12          for c in range(len(class_names)):
13              if r != c and cm[r, c] > 0:
14                  errors.append({
15                      'Asli': class_names[r],
16                      'Diprediksi': class_names[c],
17                      'Jumlah': cm[r, c],
18                      'Persentase': (cm[r, c] / cm[r].sum()) * 100
19                  })
```

Kode Program 4.9 memperlihatkan dua analisis reliabilitas secara berurutan, yaitu perhitungan skor Cohen’s Kappa per metode untuk menilai kesepakatan prediksi, serta ekstraksi kesalahan off-diagonal confusion matrix untuk mengidentifikasi pola misklasifikasi yang paling dominan.

---

### 4.5.3 Validasi Signifikansi Statistik (Kruskal-Wallis)

Perbedaan numerik antar metode yang teramati pada Tabel 4.17 belum cukup untuk menyimpulkan adanya perbedaan yang bermakna secara ilmiah. Untuk itu, diterapkan **uji Kruskal-Wallis** — uji non-parametrik yang tidak mengasumsikan normalitas distribusi — pada hasil **5-Fold Stratified Cross-Validation** menggunakan 20% subsampel (302.774 sampel). Hasil per-fold disajikan pada Tabel 4.23.

> **[TABEL 4.23 — Hasil Cross-Validation 5-Fold per Metode]**

| Fold | NSGA-II F1 | NSGA-II Time (s) | TPE F1 | TPE Time (s) | Random F1 | Random Time (s) |
|---|---|---|---|---|---|---|
| 1 | 0,8490 | 0,2399 | 0,8453 | 0,1414 | 0,8454 | 0,2433 |
| 2 | 0,8487 | 0,2423 | 0,8530 | 0,1413 | 0,8523 | 0,2479 |
| 3 | 0,8400 | 0,2407 | 0,8442 | 0,1406 | 0,8426 | 0,2487 |
| 4 | 0,8400 | 0,2397 | 0,8497 | 0,1417 | 0,8435 | 0,2473 |
| 5 | 0,8443 | 0,2394 | 0,8451 | 0,1427 | 0,8435 | 0,2495 |
| **Mean** | **0,8444** | **0,2404** | **0,8475** | **0,1415** | **0,8455** | **0,2473** |

Tabel 4.23 memperlihatkan stabilitas yang tinggi pada kedua metrik lintas fold. Variasi F1-Score antar fold sangat kecil untuk semua metode (rentang ~0,84–0,85), menunjukkan bahwa model tidak sensitif terhadap partisi data tertentu — indikator generalisasi yang baik. Pada kolom waktu inferensi, TPE secara konsisten mencatat waktu yang jauh lebih singkat di setiap fold (~0,14 detik versus ~0,24 detik untuk kedua metode lainnya), mengkonfirmasi bahwa keunggulan latensi bukan merupakan artefak pengukuran tunggal melainkan properti sistemik dari model TPE yang lebih ringan.

> **[GAMBAR 4.12 — Boxplot Cross-Validation F1-Score dan Inference Time]**
> *Deskripsi: Dua boxplot berdampingan yang menampilkan distribusi F1-Score dan Inference Time dari 5-fold CV untuk ketiga metode, memperlihatkan sebaran, median, dan potensi overlap. File referensi: `cv_stats_boxplot.png`.*

Gambar 4.12 menerjemahkan data tabel ke dalam representasi visual distribusi melalui boxplot. Pada panel F1-Score, *interquartile range* (IQR) ketiga metode tumpang tindih secara substansial — secara intuitif mengisyaratkan bahwa perbedaannya tidak signifikan. Sebaliknya, pada panel Inference Time, box TPE terposisi sepenuhnya di bawah box NSGA-II dan Random tanpa overlap sama sekali, memberikan bukti visual yang kuat bahwa keunggulan kecepatan TPE bersifat konsisten dan bukan kebetulan.

Konfirmasi formal melalui uji Kruskal-Wallis disajikan pada Tabel 4.24.

> **[TABEL 4.24 — Hasil Uji Kruskal-Wallis]**

| Metrik yang Diuji | H-statistic | p-value | Hasil | Interpretasi |
|---|---|---|---|---|
| **F1-Score** | 1,8600 | 0,3946 | **TIDAK Signifikan** (p > 0,05) | Performa deteksi ketiga metode dianggap **setara secara statistik** |
| **Inference Time** | 12,5000 | 0,0019 | **Signifikan** (p < 0,05) | Terdapat perbedaan nyata; **TPE adalah pemenang statistik** untuk kecepatan |

Tabel 4.24 menyajikan hasil uji Kruskal-Wallis yang menjadi landasan kesimpulan statistik penelitian ini. Untuk F1-Score, H-statistic yang rendah (1,86) menghasilkan p-value = 0,3946, jauh melampaui ambang batas signifikansi α = 0,05. Artinya, **hipotesis nol tidak dapat ditolak**: tidak terdapat bukti yang cukup untuk menyatakan bahwa salah satu metode lebih unggul dalam kualitas deteksi. Temuan ini memiliki implikasi praktis yang penting — pengguna dapat memilih metode HPO berdasarkan pertimbangan lain (kecepatan, kemudahan implementasi) tanpa mengorbankan akurasi.

Untuk Inference Time, H-statistic yang tinggi (12,50) menghasilkan p-value = 0,0019, jauh di bawah α = 0,05. Artinya, **hipotesis nol ditolak**: terdapat perbedaan yang signifikan secara statistik dalam kecepatan inferensi antar metode. Dengan rata-rata waktu inferensi TPE (0,1415 detik) yang hampir separuh dari NSGA-II (0,2404 detik) dan Random (0,2473 detik), **TPE dinyatakan sebagai pemenang statistik untuk efisiensi komputasi**.

Kombinasi kedua temuan ini menghasilkan rekomendasi yang terarah:
1. **Untuk skenario yang mengutamakan kualitas deteksi**, ketiga metode dapat digunakan secara bergantian karena menghasilkan performa yang setara secara statistik.
2. **Untuk skenario *real-time* yang sensitif terhadap latensi**, TPE menjadi pilihan optimal karena menghasilkan model yang secara signifikan lebih cepat tanpa mengorbankan akurasi deteksi.

Kode Program 4.10 menampilkan proses cross-validation 5-fold dan uji statistik Kruskal-Wallis untuk validasi signifikansi perbedaan antar metode.

> **[KODE PROGRAM 4.10 — Cross-Validation & Uji Kruskal-Wallis]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 16*

```
 1  N_FOLDS = 5
 2  X_cv, _, y_cv, _, w_cv, _ = train_test_split(
 3      X_train, y_train, sample_weights_train,
 4      train_size=0.20, stratify=y_train, random_state=42
 5  )
 6
 7  for name, obj in target_models.items():
 8      base_model = obj if not isinstance(obj, dict) else obj['model']
 9      params = base_model.get_params()
10      cv_model = xgb.XGBClassifier(**params)
11      skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
12
13      for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv)):
14          X_tr, X_va = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
15          y_tr, y_va = y_cv[tr_idx], y_cv[va_idx]
16          cv_model.fit(X_tr, y_tr, sample_weight=w_cv[tr_idx], verbose=False)
17
18          start_time = time.perf_counter()
19          y_pred = cv_model.predict(X_va)
20          end_time = time.perf_counter()
21          scores_f1.append(f1_score(y_va, y_pred, average="macro"))
22          scores_time.append(end_time - start_time)
23
24  stat, p = kruskal(*[cv_results[m]["f1_scores"] for m in cv_results])
```

Kode Program 4.10 menunjukkan kerangka validasi statistik yang lengkap: performa model diuji pada skema Stratified 5-Fold untuk memperoleh distribusi skor lintas fold, kemudian diuji dengan Kruskal-Wallis guna menilai signifikansi perbedaan antar metode.


---

## 4.6 Interpretasi Transparansi Model

Di luar pencapaian performa prediktif, pemahaman terhadap mekanisme internal model merupakan aspek penting dalam pengembangan sistem *trustworthy AI* di domain keamanan siber. Sub-bab ini menguraikan dua dimensi transparansi model: pertama, analisis pengaruh relatif setiap hiperparameter terhadap kedua fungsi tujuan (F1-Score dan Latency) menggunakan teknik surrogate model; kedua, analisis kepentingan fitur (*feature importance*) berbasis metrik Gain dari XGBoost yang mengidentifikasi atribut-atribut lalu lintas jaringan yang paling berkontribusi dalam keputusan klasifikasi. Kedua analisis ini memberikan wawasan praktis bagi praktisi keamanan jaringan maupun peneliti yang ingin melakukan penyesuaian hiperparameter secara terarah.

### 4.6.1 Analisis Pengaruh Hiperparameter (Surrogate Model)

Untuk menganalisis pengaruh relatif setiap hiperparameter terhadap performa model, diterapkan teknik **Random Forest Surrogate Model** — di mana model surrogate dilatih untuk memprediksi metrik performa berdasarkan vektor hiperparameter, kemudian *feature importance*-nya digunakan sebagai estimasi pengaruh setiap hiperparameter.

Kode Program 4.11 menampilkan fungsi inti analisis importance menggunakan Random Forest Regressor sebagai surrogate model.

> **[KODE PROGRAM 4.11 — Hyperparameter Importance (Surrogate RF)]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 11*

```
 1  from sklearn.ensemble import RandomForestRegressor
 2
 3  def get_rf_importance(study, objective_name, objective_index):
 4      trials = [t for t in study.trials
 5                if t.state == optuna.trial.TrialState.COMPLETE]
 6      data = []
 7      for t in trials:
 8          row = {k: v for k, v in t.params.items()
 9                 if isinstance(v, (int, float))}
10          row['target'] = t.values[objective_index]
11          data.append(row)
12
13      df = pd.DataFrame(data)
14      X = df.drop(columns='target')
15      y = df['target']
16
17      model = RandomForestRegressor(
18          n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
19      )
20      model.fit(X, y)
21      importances = pd.Series(model.feature_importances_, index=X.columns)
22      return importances.sort_values(ascending=False)
```

Kode Program 4.11 mengimplementasikan pendekatan surrogate model, yakni menggunakan Random Forest Regressor untuk memodelkan hubungan antara hiperparameter dan nilai objektif, lalu menurunkan tingkat kepentingan masing-masing hiperparameter dari skor feature importance model surrogate.

#### Pengaruh Hiperparameter terhadap F1-Score

> **[TABEL 4.25 — Importance Hiperparameter terhadap F1-Score]**

| Peringkat | TPE (Importance) | NSGA-II (Importance) | Random (Importance) |
|---|---|---|---|
| 1 | `learning_rate` (0,2809) | `learning_rate` (0,6934) | `learning_rate` (0,6550) |
| 2 | `subsample` (0,1981) | `subsample` (0,0941) | `max_depth` (0,0739) |
| 3 | `gamma` (0,1279) | `colsample_bytree` (0,0390) | `subsample` (0,0714) |

Tabel 4.25 mengungkap konsistensi yang mencolok: **`learning_rate` menduduki peringkat pertama** dalam mempengaruhi F1-Score di **ketiga metode** tanpa terkecuali, dengan importance berkisar dari 0,2809 (TPE) hingga 0,6934 (NSGA-II). Temuan ini mengkonfirmasi bahwa laju pembelajaran merupakan pengendali utama keseimbangan antara *bias* dan *variance* model. Pada TPE, importance tersebar lebih merata antar hiperparameter, mengindikasikan interaksi yang lebih kompleks antar parameter dalam metode Bayesian. `subsample` secara konsisten berada di peringkat 2–3, menunjukkan perannya yang penting dalam regularisasi melalui teknik *stochastic gradient boosting*.

> **[GAMBAR 4.13 — Bar Chart Importance Hiperparameter terhadap F1 (TPE)]**
> *Deskripsi: Horizontal bar chart yang menampilkan kontribusi relatif setiap hiperparameter terhadap variasi F1-Score pada metode TPE. Panjang bar merepresentasikan proporsi pengaruh. File referensi: `importance_f1_tpe.png`.*

Gambar 4.13 memvisualisasikan distribusi importance untuk metode TPE dalam format bar chart horizontal. Distribusi yang relatif merata antara `learning_rate` (0,28), `subsample` (0,20), dan `gamma` (0,13) menunjukkan bahwa TPE mengeksplorasi ruang hiperparameter secara lebih menyeluruh, tidak terlalu bergantung pada satu parameter tunggal. Pola ini konsisten dengan mekanisme Bayesian TPE yang secara aktif memodelkan interaksi antar hiperparameter.

> **[GAMBAR 4.14 — Bar Chart Importance Hiperparameter terhadap F1 (NSGA-II)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter terhadap F1-Score pada metode NSGA-II, menunjukkan dominasi learning_rate yang lebih kuat. File referensi: `importance_f1_nsga-ii.png`.*

Gambar 4.14 menunjukkan kontras yang tajam dengan TPE: `learning_rate` mendominasi hingga 69,34% dari total importance pada NSGA-II, sementara hiperparameter lainnya memiliki pengaruh yang relatif marginal. Konsentrasi ini mengisyaratkan bahwa variasi performa antar trial NSGA-II terutama didorong oleh perbedaan learning rate, sementara parameter lain berperan sebagai penyesuaian halus (*fine-tuning*).

> **[GAMBAR 4.15 — Bar Chart Importance Hiperparameter terhadap F1 (Random)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter terhadap F1-Score pada metode Random, dengan pola serupa NSGA-II di mana learning_rate mendominasi. File referensi: `importance_f1_random.png`.*

Gambar 4.15 memperlihatkan pola yang serupa dengan NSGA-II di mana `learning_rate` mendominasi (0,6550). Kesamaan pola antara NSGA-II dan Random — yang keduanya mengeksplorasi ruang pencarian secara lebih luas — menunjukkan bahwa ketika eksplorasi bersifat global, variasi learning rate menjadi pembeda utama antar konfigurasi. Sebaliknya, TPE yang lebih terfokus (*exploitative*) menemukan region di mana learning rate sudah relatif terkontrol, sehingga parameter lain menjadi lebih berpengaruh.

#### Pengaruh Hiperparameter terhadap Latency

> **[TABEL 4.26 — Importance Hiperparameter terhadap Latency]**

| Peringkat | TPE (Importance) | NSGA-II (Importance) | Random (Importance) |
|---|---|---|---|
| 1 | `n_estimators` (0,7534) | `n_estimators` (0,6570) | `n_estimators` (0,6713) |
| 2 | `reg_alpha` (0,0655) | `max_depth` (0,1343) | `max_depth` (0,1277) |
| 3 | `subsample` (0,0318) | `reg_alpha` (0,0569) | `gamma` (0,0523) |

Tabel 4.26 menampilkan hasil analisis importance untuk objektif latensi. Konsistensi yang ditemukan bahkan lebih kuat dibanding analisis F1: **`n_estimators` mendominasi pengaruh terhadap latensi di ketiga metode** dengan importance 0,66–0,75. Temuan ini mengkonfirmasi bahwa kompleksitas inferensi bersifat linear terhadap jumlah estimator: $T_{inference} \propto K \times D$, di mana $K$ adalah jumlah pohon dan $D$ rata-rata kedalaman. `max_depth` berada di peringkat kedua pada NSGA-II dan Random (importance ~0,13), karena kedalaman pohon menentukan jumlah perbandingan yang diperlukan per pohon.

> **[GAMBAR 4.16 — Bar Chart Importance Hiperparameter terhadap Latency (TPE)]**
> *Deskripsi: Horizontal bar chart kontribusi relatif setiap hiperparameter terhadap variasi latensi inferensi pada metode TPE. File referensi: `importance_time_tpe.png`.*

Gambar 4.16 memvisualisasikan dominasi `n_estimators` yang sangat kuat (75,34%) pada metode TPE. Hiperparameter lain nyaris tidak berpengaruh terhadap latensi, mengindikasikan bahwa dalam konfigurasi TPE — yang cenderung menghasilkan model dengan jumlah pohon bervariasi — jumlah pohon menjadi satu-satunya faktor yang menentukan kecepatan inferensi secara signifikan.

> **[GAMBAR 4.17 — Bar Chart Importance Hiperparameter terhadap Latency (NSGA-II)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter terhadap Latency pada metode NSGA-II, dengan `n_estimators` dominan dan `max_depth` sebagai faktor sekunder. File referensi: `importance_time_nsga-ii.png`.*

Gambar 4.17 menunjukkan distribusi yang sedikit lebih tersebar dibanding TPE, dengan `max_depth` memberikan kontribusi 13,43% pada NSGA-II. Hal ini logis karena NSGA-II cenderung menghasilkan model dengan kedalaman yang lebih bervariasi, sehingga perbedaan kedalaman antar trial memberikan dampak latensi yang lebih terukur.

> **[GAMBAR 4.18 — Bar Chart Importance Hiperparameter terhadap Latency (Random)]**
> *Deskripsi: Horizontal bar chart importance hiperparameter terhadap Latency pada metode Random, menunjukkan pola serupa dengan NSGA-II. File referensi: `importance_time_random.png`.*

Gambar 4.18 memperlihatkan pola yang konsisten dengan NSGA-II, di mana `n_estimators` (67,13%) dan `max_depth` (12,77%) merupakan dua faktor utama penentu latensi. Konsistensi pola ini lintas ketiga metode memperkuat validitas temuan dan memberikan panduan praktis bagi praktisi: **untuk mengurangi latensi inferensi pada model XGBoost, prioritaskan pengurangan jumlah pohon terlebih dahulu, kemudian kedalaman pohon**.

---

### 4.6.2 Analisis Kepentingan Fitur (XGBoost Feature Importance)

Analisis kepentingan fitur dilakukan menggunakan metrik **Gain** dari XGBoost, yang mengukur peningkatan rata-rata kualitas split (*loss reduction*) yang dikontribusikan oleh setiap fitur di seluruh pohon dalam ensemble.

Kode Program 4.12 menampilkan proses ekstraksi feature importance berbasis Gain dari model XGBoost yang telah dilatih.

> **[KODE PROGRAM 4.12 — Feature Importance XGBoost (Gain)]**
> *Sumber: `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 13*

```
 1  feature_importance_data = {}
 2
 3  for name, model in trained_models.items():
 4      booster = model.get_booster()
 5      importance_dict = booster.get_score(importance_type='gain')
 6      all_features = booster.feature_names
 7      full_importance = {f: importance_dict.get(f, 0.0)
 8                         for f in all_features}
 9      feature_importance_data[name] = full_importance
10
11  global_scores = {}
12  for imp in feature_importance_data.values():
13      for f, v in imp.items():
14          global_scores[f] = global_scores.get(f, 0) + v
15
16  top_20_global = [x[0] for x in sorted(
17      global_scores.items(), key=lambda x: x[1], reverse=True)[:20]]
```

Kode Program 4.12 memperlihatkan cara menghimpun feature importance berbasis Gain dari setiap model, kemudian mengagregasikannya menjadi skor global sehingga fitur-fitur paling berpengaruh dapat diidentifikasi secara konsisten lintas metode optimasi.

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

Tabel 4.27 menyajikan 10 fitur terpenting berdasarkan rata-rata Gain dari seluruh model ketiga metode. Dua temuan utama terlihat mencolok: pertama, **`MIN_TTL` dan `MAX_TTL` mendominasi secara absolut** dengan gain yang jauh melampaui fitur lainnya (14.359 dan 11.645 versus 3.786 untuk peringkat ketiga), mengkonfirmasi bahwa TTL merupakan diskriminator utama karena pola anomali TTL dapat membedakan lalu lintas *legitimate* dari *malicious*. Kedua, **fitur ukuran paket** (`MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT`) menempati posisi penting, karena banyak serangan scanning dan probing menggunakan paket-paket berukuran kecil (SYN probes, ICMP echo).

Perbandingan kepentingan fitur antar metode disajikan pada Tabel 4.28 untuk mengidentifikasi konsistensi lintas model.

> **[TABEL 4.28 — Top 5 Fitur Terpenting per Metode]**

| Peringkat | NSGA-II (Gain) | TPE (Gain) | Random (Gain) |
|---|---|---|---|
| 1 | MIN_TTL (3.941,30) | MIN_TTL (5.415,10) | MIN_TTL (5.002,60) |
| 2 | MAX_TTL (3.029,74) | MAX_TTL (4.595,72) | MAX_TTL (4.019,58) |
| 3 | MIN_IP_PKT_LEN (1.360,91) | SHORTEST_FLOW_PKT (1.036,94) | MIN_IP_PKT_LEN (1.444,39) |
| 4 | SHORTEST_FLOW_PKT (824,66) | MIN_IP_PKT_LEN (980,91) | DNS_QUERY_TYPE (785,55) |
| 5 | DNS_QUERY_TYPE (751,83) | DNS_QUERY_TYPE (947,20) | SHORTEST_FLOW_PKT (726,41) |

Tabel 4.28 mengkonfirmasi konsistensi peringkat fitur lintas metode: `MIN_TTL` dan `MAX_TTL` secara unanimous menduduki posisi pertama dan kedua, diikuti oleh `MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT`, dan `DNS_QUERY_TYPE` yang saling bertukar posisi 3–5. Stabilitas peringkat ini menunjukkan bahwa model-model yang dihasilkan oleh metode optimasi berbeda tetap mengandalkan sinyal fitur yang sama untuk keputusan klasifikasi — bukti kuat bahwa pola diskriminatif yang dipelajari bersifat genuine dan bukan artefak dari proses pencarian hiperparameter tertentu.

> **[GAMBAR 4.19 — Bar Chart Perbandingan Feature Importance Ketiga Metode]**
> *Deskripsi: Grouped horizontal bar chart yang menampilkan fitur terpenting untuk setiap metode secara berdampingan, memungkinkan identifikasi konsistensi dan perbedaan antar model. File referensi: `feature_importance_bar_comparison.png`.*

Gambar 4.19 menyajikan perbandingan visual importance fitur antar metode dalam format *grouped bar chart*. Representasi ini memperjelas bahwa meskipun magnitude gain bervariasi antar metode (TPE cenderung lebih tinggi karena jumlah pohon lebih sedikit sehingga tiap split memberikan kontribusi rata-rata lebih besar), **urutan peringkat relatif sangat konsisten**. Pola ini memperkuat kepercayaan bahwa model telah mempelajari representasi fitur yang bermakna secara domain.

> **[GAMBAR 4.20 — Heatmap Feature Importance Ketiga Metode]**
> *Deskripsi: Heatmap yang menampilkan importance score untuk seluruh 49 fitur di ketiga metode, memungkinkan analisis pola kepentingan fitur secara holistik. File referensi: `feature_importance_heatmap_comparison.png`.*

Gambar 4.20 memberikan pandangan holistik terhadap seluruh 49 fitur melalui format heatmap. Dari visualisasi ini teridentifikasi bahwa sebagian besar fitur memiliki kontribusi yang relatif kecil (area berwarna pudar), sementara segelintir fitur kunci (TTL, ukuran paket, DNS) mendominasi proses keputusan (area berwarna intens). Pola *sparse importance* ini konsisten dengan prinsip parsimoni dalam *machine learning* — model yang efektif sering kali mengandalkan sejumlah kecil fitur yang sangat informatif.

**Interpretasi domain dari fitur-fitur terpenting:**

1. **`MIN_TTL` dan `MAX_TTL` (Time-To-Live):** TTL pada paket IP menentukan jumlah hop maksimum sebelum paket didiscard. Serangan jaringan sering menghasilkan pola TTL anomali — misalnya, *IP spoofing* menghasilkan TTL yang tidak konsisten dengan jarak topologis sebenarnya, dan *traceroute-based reconnaissance* secara sengaja memanipulasi TTL. Dominasi fitur TTL mengkonfirmasi relevansinya sebagai diskriminator utama antara lalu lintas normal dan *malicious*.

2. **`MIN_IP_PKT_LEN` dan `SHORTEST_FLOW_PKT`:** Ukuran paket minimum dalam suatu flow sangat informatif karena serangan scanning dan probing umumnya menggunakan paket berukuran kecil (SYN packets ~60 bytes, ICMP echo ~64 bytes) yang berbeda dari trafik aplikasi normal yang biasanya berukuran lebih besar.

3. **`DNS_QUERY_TYPE`:** Tipe query DNS merupakan indikator penting untuk mendeteksi serangan DNS-based seperti *DNS tunneling*, *DNS amplification*, dan *data exfiltration* yang menggunakan tipe query tidak lazim seperti TXT, NULL, dan MX.

4. **`L7_PROTO` dan `L4_DST_PORT`:** Protokol aplikasi (Layer 7) dan port tujuan (Layer 4) membantu membedakan jenis layanan yang menjadi target, di mana port-port tertentu yang terkait layanan rentan (SMB/445, RDP/3389, SSH/22) sering kali menjadi target eksploitasi.


---

## 4.7 Implementasi Simulasi Prototipe NIDS

Validasi akhir suatu model klasifikasi tidak cukup hanya melalui evaluasi metrik pada data uji statis, melainkan perlu didemonstrasikan dalam konteks operasional yang mendekati skenario penerapan sesungguhnya. Sub-bab ini mendokumentasikan implementasi prototipe *Network Intrusion Detection System* (NIDS) berbasis web yang dibangun menggunakan framework Streamlit, di mana model-model Pareto-optimal yang dihasilkan dari proses optimasi diintegrasikan ke dalam pipeline deteksi intrusi secara *near real-time*. Pembahasan mencakup dua aspek utama: desain antarmuka dashboard beserta arsitektur pipeline pemrosesan data, serta hasil pengujian pada dua skenario simulasi — baseline dengan lalu lintas normal untuk mengukur tingkat *false alarm*, dan injection dengan campuran serangan untuk mengevaluasi kemampuan deteksi aktual. Prototipe ini berfungsi sebagai *proof of concept* yang menjembatani temuan eksperimental dengan kebutuhan operasional praktis.

### 4.7.1 Tampilan Antarmuka Dashboard

Validasi akhir dari model klasifikasi memerlukan pengujian dalam konteks yang mendekati skenario penerapan sesungguhnya. Untuk tujuan ini, dikembangkan prototipe **Network Intrusion Detection System (NIDS)** berbasis web menggunakan framework **Streamlit** yang mensimulasikan proses deteksi intrusi secara *near real-time*. Prototipe ini memuat model-model Pareto-optimal yang telah diekspor dari tahap pelatihan dan menjalankan inferensi pada data simulasi paket demi paket, meniru operasional *inline* NIDS pada jaringan aktif.

> **[GAMBAR 4.21 — Tampilan Dashboard Utama NIDS (State Awal)]**
> *Deskripsi: Screenshot halaman utama dashboard yang menampilkan panel kontrol pada sidebar kiri (pemilihan model, skenario, dan parameter simulasi) serta area dashboard utama pada sisi kanan (indikator status, gauge latensi, dan tabel log deteksi) dalam state awal sebelum simulasi dijalankan. Ambil screenshot dari aplikasi Streamlit yang telah di-deploy.*

Gambar 4.21 menampilkan tampilan awal dashboard NIDS yang terdiri dari dua area utama: sidebar kiri berfungsi sebagai panel kontrol yang memuat seluruh parameter konfigurasi simulasi, sementara area utama di sisi kanan menyajikan indikator status, metrik latensi, dan tabel riwayat deteksi. Desain antarmuka mengikuti prinsip *progressive disclosure* dengan pengaturan teknis di sidebar agar fokus pengguna tetap pada informasi deteksi di area utama.

**Arsitektur Pipeline Aplikasi:**

Pipeline pemrosesan data dalam prototipe terdiri dari 5 tahap sekuensial yang dieksekusi per paket data:

1. **Pemuatan Artefak** — Model XGBoost (`.json`), `StandardScaler` (`.pkl`), dan `LabelEncoder` (`.pkl`) dimuat ke memori dengan mekanisme *lazy loading* dan *caching* untuk efisiensi.
2. **Akuisisi & Pra-proses** — Data disimulasikan melalui streaming satu baris per iterasi dari `simulation_data.csv`, dengan fitur diurutkan sesuai `scaler.feature_names_in_` untuk menjamin konsistensi.
3. **Inferensi & Timing** — Prediksi probabilitas (`predict_proba`) dieksekusi dengan pengukuran latensi presisi tinggi menggunakan `time.perf_counter_ns` (resolusi nanosecond).
4. **Dekoding Keputusan** — Kelas integer (0–3) dipetakan ke label yang bermakna (Normal, DoS, Probe, Malware) dengan *confidence score* dari probabilitas tertinggi.
5. **Visualisasi & UI** — Indikator status, gauge latensi, kartu peringatan, dan tabel log diperbarui secara *reactive* sesuai hasil prediksi.

Kode Program 4.13 menampilkan pipeline inferensi inti pada dashboard, mencakup pemuatan artefak model dan fungsi prediksi dengan pengukuran latensi presisi tinggi.

> **[KODE PROGRAM 4.13 — Pipeline Inferensi Dashboard NIDS]**
> *Sumber: `streamlit_app/app.py`*

```
 1  @st.cache_resource
 2  def load_scaler():
 3      if SCALER_PATH.exists():
 4          return joblib.load(SCALER_PATH)
 5      return None
 6
 7  class InferenceEngine:
 8      def load_model(self, filename):
 9          path = MODELS_DIR / filename
10          self.model = xgb.XGBClassifier()
11          self.model.load_model(str(path))
12
13      def predict(self, data):
14          t0 = time.perf_counter_ns()
15          probs = self.model.predict_proba(data)[0]
16          t1 = time.perf_counter_ns()
17
18          idx = int(np.argmax(probs))
19          label = LABEL_MAP.get(idx, "Unknown")
20          confidence = float(probs[idx])
21          latency_ms = (t1 - t0) / 1e6
22          return idx, label, confidence, latency_ms
23
24  def prepare_features(row_df):
25      features = row_df.drop(columns=["Label_True"], errors="ignore")
26      if EXPECTED_FEATURES:
27          features = features[EXPECTED_FEATURES]
28      return features.values
```

Kode Program 4.13 menggambarkan fondasi pipeline inferensi pada dashboard NIDS, mulai dari pemuatan artefak model secara efisien, prediksi probabilitas dengan pengukuran latensi presisi tinggi, hingga penyiapan urutan fitur agar konsisten dengan skema pelatihan.

Detail komponen antarmuka didokumentasikan pada Tabel 4.29 dan 4.30.

> **[TABEL 4.29 — Komponen Panel Kontrol (Sidebar)]**

| No | Komponen | Tipe Widget | Deskripsi |
|---|---|---|---|
| 1 | Metode Optimasi | `selectbox` | Pemilihan metode HPO: TPE, Random, atau NSGA-II |
| 2 | Model Pareto | `selectbox` | Pemilihan model Pareto-optimal spesifik berdasarkan katalog |
| 3 | Skenario Pengujian | `radio` | Baseline (hanya trafik Normal) vs Injection (campuran serangan) |
| 4 | Jumlah Data Simulasi | `selectbox` | Opsi 50, 100, atau 150 paket per sesi simulasi |
| 5 | Acak Data | `checkbox` | Randomisasi urutan baris untuk variasi antar percobaan |
| 6 | Kecepatan Simulasi | `slider` | Interval antar paket: 0,5 – 5,0 detik (default: 2,0) |
| 7 | Auto-stop | `checkbox` | Jeda otomatis saat serangan pertama terdeteksi (default: aktif) |
| 8 | Tombol START/STOP | `button` | Memulai atau menghentikan streaming kontinyu |
| 9 | Tombol NEXT | `button` | Maju satu paket (mode step-by-step untuk analisis detail) |
| 10 | Tombol RESET | `button` | Mengembalikan seluruh state, log, dan penghitung ke kondisi awal |

Tabel 4.29 mendokumentasikan 10 komponen kontrol yang tersedia pada sidebar. Desain ini memberikan fleksibilitas pengujian yang tinggi: pengguna dapat membandingkan performa model dari metode HPO yang berbeda, menguji skenario normal versus serangan, mengatur kecepatan simulasi untuk observasi detail, dan menggunakan mode step-by-step untuk menganalisis respons model terhadap setiap paket secara individual. Fitur auto-stop secara khusus berguna untuk mengidentifikasi momen pertama model mendeteksi ancaman.

> **[TABEL 4.30 — Komponen Area Dashboard Utama]**

| No | Komponen | Deskripsi |
|---|---|---|
| 1 | Kartu Status | Indikator visual: Normal (✅ hijau) atau Attack (🚨 merah) dengan animasi CSS *pulse* |
| 2 | Metrik Latensi | Angka latensi inferensi saat ini dan rata-rata kumulatif |
| 3 | Metrik Skenario | False Positive count (Baseline) atau Packet count (Injection) |
| 4 | Summary Bar | Penghitung berjalan: True Positive / True Negative / False Positive / False Negative |
| 5 | Gauge Latensi | Plotly gauge (0–100 ms) dengan garis threshold merah pada 70 ms |
| 6 | Kartu Alert | Kartu animasi kontekstual: TP (merah), FP (oranye), FN (ungu) |
| 7 | Tabel Log Deteksi | Tabel HTML dengan baris berwarna: merah (serangan), oranye (FP), ungu (FN), putih (normal) |
| 8 | Grafik Ringkasan | Donut chart (TP/TN/FP/FN), bar chart prediksi, line chart latensi — muncul pasca-simulasi |

Tabel 4.30 merinci 8 komponen visualisasi pada area dashboard utama. Setiap komponen dirancang untuk menyampaikan informasi spesifik: kartu status memberikan *situational awareness* instan melalui kode warna, gauge latensi memantau performa real-time terhadap ambang batas operasional, kartu alert menyajikan notifikasi kontekstual yang berbeda untuk setiap jenis outcome deteksi (TP/FP/FN), dan tabel log menyimpan riwayat komprehensif dengan *color coding* yang memudahkan identifikasi pola. Grafik ringkasan yang muncul setelah simulasi berakhir menyediakan analisis retrospektif melalui distribusi outcome dan profil latensi temporal.

**Katalog Model yang Tersedia:**

Dashboard memuat **7 model Pareto-optimal** dari ketiga metode, yang disimpan dalam format JSON XGBoost dan dikatalogkan pada Tabel 4.31.

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

Tabel 4.31 menampilkan metadata ketujuh model yang dapat dipilih pengguna melalui antarmuka dashboard. Ketujuh model merepresentasikan titik-titik *trade-off* yang berbeda pada Pareto front — dari model dengan akurasi tertinggi (F1=0,8648, latensi 5,4 µs) hingga model tercepat (F1=0,8565, latensi 3,7 µs). Ketersediaan beragam model ini memungkinkan pengguna menyesuaikan karakteristik NIDS sesuai kebutuhan spesifik: model dengan F1 tinggi untuk lingkungan yang mengutamakan akurasi deteksi, atau model dengan latensi rendah untuk lingkungan yang memerlukan respons tercepat.

Data simulasi (`simulation_data.csv`) berisi 2.000 sampel yang telah distandardisasi, dengan distribusi: 1.896 Normal (94,8%), 28 DoS (1,4%), 12 Probe (0,6%), dan 64 Malware (3,2%). Proporsi ini merepresentasikan distribusi trafik jaringan yang realistis di mana lalu lintas normal mendominasi dengan insiden serangan yang sporadis.

---

### 4.7.2 Hasil Pengujian Skenario Simulasi

#### Skenario 1: Baseline (Lalu Lintas Normal)

Skenario baseline dirancang untuk mengukur **kinerja dasar** model dengan hanya menggunakan data berlabel Normal (kelas 0). Tujuan utamanya adalah mengukur latensi inferensi baseline dan mengidentifikasi adanya **False Positive** — situasi di mana model salah memicu alarm pada trafik yang sebenarnya normal.

> **[GAMBAR 4.22 — Screenshot Dashboard saat Skenario Baseline Berjalan]**
> *Deskripsi: Screenshot yang menampilkan dashboard dengan indikator hijau stabil (Normal ✅), gauge latensi berada di zona hijau, penghitung FP tetap nol, dan tabel log menunjukkan seluruh prediksi adalah "Normal" dengan confidence tinggi. Ambil screenshot dari aplikasi Streamlit.*

Gambar 4.22 mendokumentasikan jalannya skenario baseline pada dashboard. Indikator status mempertahankan warna hijau sepanjang simulasi, gauge latensi berada di zona aman (di bawah threshold 70 ms), dan seluruh baris pada tabel log menampilkan prediksi "Normal" — menunjukkan bahwa model berhasil mengklasifikasikan seluruh trafik normal dengan benar. Hasil ini konsisten dengan skor precision Normal = 1,0000 yang teramati pada evaluasi test set sebelumnya (Tabel 4.18–4.20), mengkonfirmasi bahwa model memiliki tingkat *false alarm* yang sangat rendah.

Ekspektasi pengujian pada skenario baseline:
- Seluruh prediksi menghasilkan label "Normal" dengan confidence tinggi
- Indikator status tetap berwarna hijau (✅) tanpa fluktuasi
- Penghitung False Positive tetap pada angka 0
- Latensi inferensi stabil dan berada di bawah ambang batas operasional

#### Skenario 2: Injection (Campuran Serangan)

Skenario injection merupakan pengujian inti yang menggunakan dataset lengkap — termasuk sampel serangan DoS, Probe, dan Malware — untuk mengevaluasi **kemampuan deteksi** dan **responsivitas** model terhadap ancaman.

Kode Program 4.14 menampilkan logika inti simulasi loop pada dashboard, mencakup akuisisi data, inferensi, dan klasifikasi event (TP/TN/FP/FN).

> **[KODE PROGRAM 4.14 — Simulasi Loop & Klasifikasi Event]**
> *Sumber: `streamlit_app/app.py`*

```
 1  row_data = stream.iloc[[st.session_state.idx]]
 2  true_label_idx = int(row_data["Label_True"].iloc[0])
 3  true_label = LABEL_MAP.get(true_label_idx, "Unknown")
 4
 5  input_data = prepare_features(row_data)
 6  pred_idx, pred_label, conf, lat = st.session_state.engine.predict(input_data)
 7  st.session_state.latency_history.append(lat)
 8
 9  is_true_attack = true_label_idx > 0
10  is_pred_attack = pred_idx > 0
11
12  if is_pred_attack and is_true_attack:
13      event_type = "tp"
14      st.session_state.tp_count += 1
15  elif not is_pred_attack and not is_true_attack:
16      event_type = "tn"
17      st.session_state.tn_count += 1
18  elif is_pred_attack and not is_true_attack:
19      event_type = "fp"
20      st.session_state.fp_count += 1
21  elif not is_pred_attack and is_true_attack:
22      event_type = "fn"
23      st.session_state.fn_count += 1
24
25  if auto_stop and is_pred_attack:
26      st.session_state.run = False
```

Kode Program 4.14 menjelaskan alur eksekusi simulasi per paket pada dashboard: data diambil bertahap, diprediksi oleh engine, diklasifikasikan ke outcome TP/TN/FP/FN, metrik sesi diperbarui real-time, dan simulasi dapat dihentikan otomatis saat serangan terdeteksi.

> **[GAMBAR 4.23 — Screenshot Dashboard saat Serangan Terdeteksi (True Positive)]**
> *Deskripsi: Screenshot yang menangkap momen deteksi serangan — indikator berubah merah (🚨 Attack), kartu alert merah muncul menandakan True Positive, confidence score dan label prediksi ditampilkan, serta baris tabel log berwarna merah mengidentifikasi paket serangan. Ambil screenshot dari aplikasi Streamlit.*

Gambar 4.23 menangkap momen krusial saat model berhasil mendeteksi paket serangan (*True Positive*). Transisi visual dari indikator hijau ke merah, kemunculan kartu alert animasi, dan pewarnaan baris tabel log secara kolektif memberikan *situational awareness* yang segera kepada operator NIDS. Confidence score yang ditampilkan memungkinkan operator menilai tingkat keyakinan model terhadap keputusannya, menjadi dasar untuk proses triase dan eskalasi insiden.

> **[GAMBAR 4.24 — Screenshot Grafik Ringkasan Pasca-Simulasi]**
> *Deskripsi: Screenshot yang menampilkan tiga grafik ringkasan yang muncul setelah simulasi selesai: (1) Donut chart distribusi outcome (TP/TN/FP/FN) menunjukkan proporsi deteksi yang benar dan salah, (2) Bar chart distribusi label prediksi per kelas, dan (3) Line chart latensi per paket dengan garis threshold 70 ms dan garis rata-rata. Ambil screenshot dari aplikasi Streamlit.*

Gambar 4.24 menyajikan analisis retrospektif pasca-simulasi melalui tiga grafik ringkasan. Donut chart memberikan gambaran keseluruhan proporsi outcome deteksi, memungkinkan evaluasi cepat terhadap akurasi model. Bar chart distribusi prediksi menunjukkan bagaimana model mengklasifikasikan paket-paket ke setiap kelas. Line chart latensi memetakan profil waktu inferensi per paket, menunjukkan stabilitas dan konsistensi kecepatan model sepanjang sesi simulasi — informasi yang vital untuk memastikan NIDS mampu mempertahankan throughput yang memadai dalam operasional jangka panjang.

Ekspektasi pengujian pada skenario injection:
- Indikator berubah dari hijau ke merah secara tepat saat paket serangan terdeteksi
- Kartu alert muncul sesuai konteks: merah untuk True Positive (deteksi benar), oranye untuk False Positive (alarm palsu), ungu untuk False Negative (serangan terlewat)
- Penghitung TP meningkat untuk setiap serangan yang berhasil diidentifikasi
- Fitur auto-stop menghentikan simulasi secara otomatis saat serangan pertama terdeteksi, memfasilitasi analisis deteksi individual
- Grafik latensi temporal menunjukkan stabilitas waktu inferensi lintas seluruh paket


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

### Daftar Kode Program

| No. | Nomor | Judul Kode Program | Sumber |
|---|---|---|---|
| 1 | Kode Program 4.1 | Konfigurasi Lingkungan & Verifikasi GPU | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 1 |
| 2 | Kode Program 4.2 | Pemuatan Dataset & Pemetaan Kelas | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 2 |
| 3 | Kode Program 4.3 | Pembersihan Kolom & Penanganan NaN/Infinity | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 |
| 4 | Kode Program 4.4 | Standardisasi & Pembagian Data | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 |
| 5 | Kode Program 4.5 | Perhitungan Bobot Hybrid Cost-Sensitive | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 3 |
| 6 | Kode Program 4.6 | Fungsi Objective Multi-Objective | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 4 |
| 7 | Kode Program 4.7 | Konfigurasi & Eksekusi Optimasi Optuna | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 5 |
| 8 | Kode Program 4.8 | Evaluasi Final & Retrain Model Terbaik | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 8 |
| 9 | Kode Program 4.9 | Cohen's Kappa & Analisis Pola Kesalahan | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 10B |
| 10 | Kode Program 4.10 | Cross-Validation & Uji Kruskal-Wallis | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 16 |
| 11 | Kode Program 4.11 | Hyperparameter Importance (Surrogate RF) | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 11 |
| 12 | Kode Program 4.12 | Feature Importance XGBoost (Gain) | `kode-skripsi-nf-unsw-nb15-v3 (15).ipynb` — Cell 13 |
| 13 | Kode Program 4.13 | Pipeline Inferensi Dashboard NIDS | `streamlit_app/app.py` |
| 14 | Kode Program 4.14 | Simulasi Loop & Klasifikasi Event | `streamlit_app/app.py` |

### Daftar Gambar

| No. | Nomor | Judul Gambar | File Referensi |
|---|---|---|---|
| 1 | Gambar 4.1 | Diagram Distribusi Kelas Dataset | `distribusi_dan_bobot_skripsi.png` |
| 2 | Gambar 4.2 | Dampak Pembobotan terhadap Distribusi Kelas | `impact_weighting_skripsi.png` |
| 3 | Gambar 4.3 | Grafik Riwayat Optimasi | `optimization_history_final.png` |
| 4 | Gambar 4.4 | Pareto Front Statis (Ketiga Metode) | `pareto_front_static_hd.png` |
| 5 | Gambar 4.5 | Diagram Batang Perbandingan Metrik | `metrics_grouped_bar.png` |
| 6 | Gambar 4.6 | Heatmap F1-Score per Kelas per Metode | `metrics_f1_heatmap.png` |
| 7 | Gambar 4.7 | Scatter Plot Precision vs Recall | `metrics_pr_scatter.png` |
| 8 | Gambar 4.8 | Ringkasan Recall per Kelas | `recall_summary_chart.png` |
| 9 | Gambar 4.9 | Confusion Matrix Raw | `cm_raw_heatmap.png` |
| 10 | Gambar 4.10 | Confusion Matrix Normalized | `cm_norm_heatmap.png` |
| 11 | Gambar 4.11 | Perbandingan Cohen's Kappa | `kappa_comparison.png` |
| 12 | Gambar 4.12 | Boxplot Cross-Validation | `cv_stats_boxplot.png` |
| 13 | Gambar 4.13 | Importance Hiperparameter F1 (TPE) | `importance_f1_tpe.png` |
| 14 | Gambar 4.14 | Importance Hiperparameter F1 (NSGA-II) | `importance_f1_nsga-ii.png` |
| 15 | Gambar 4.15 | Importance Hiperparameter F1 (Random) | `importance_f1_random.png` |
| 16 | Gambar 4.16 | Importance Hiperparameter Latency (TPE) | `importance_time_tpe.png` |
| 17 | Gambar 4.17 | Importance Hiperparameter Latency (NSGA-II) | `importance_time_nsga-ii.png` |
| 18 | Gambar 4.18 | Importance Hiperparameter Latency (Random) | `importance_time_random.png` |
| 19 | Gambar 4.19 | Feature Importance Bar (Perbandingan) | `feature_importance_bar_comparison.png` |
| 20 | Gambar 4.20 | Feature Importance Heatmap | `feature_importance_heatmap_comparison.png` |
| 21 | Gambar 4.21 | Dashboard Utama NIDS (State Awal) | *Screenshot dari Streamlit* |
| 22 | Gambar 4.22 | Dashboard Skenario Baseline | *Screenshot dari Streamlit* |
| 23 | Gambar 4.23 | Dashboard Serangan Terdeteksi (TP) | *Screenshot dari Streamlit* |
| 24 | Gambar 4.24 | Grafik Ringkasan Pasca-Simulasi | *Screenshot dari Streamlit* |

---

> **Catatan untuk pemindahan ke Word:**
> - Semua teks dalam blok `> **[TABEL X.X — ...]**` menandai posisi di mana tabel harus disisipkan di dokumen Word.
> - Semua teks dalam blok `> **[GAMBAR X.X — ...]**` menandai posisi di mana gambar harus disisipkan beserta caption-nya.
> - Semua teks dalam blok `> **[KODE PROGRAM X.X — ...]**` menandai posisi di mana potongan kode program harus disisipkan. Nomor baris pada kode berfungsi sebagai referensi penjelasan.
> - Setiap tabel dan gambar telah dilengkapi paragraf deskripsi/penjelasan di bawahnya yang menjelaskan isi, signifikansi, dan interpretasi.
> - File PNG referensi tersedia di arsip output notebook (`ARSIP_LENGKAP_SKRIPSI_20260212_0748.zip`, ukuran 189,29 MB).
> - Gambar 4.20–4.24 perlu diambil secara manual sebagai screenshot dari aplikasi Streamlit yang telah di-deploy.
> - Format angka menggunakan koma desimal (Indonesia) sesuai standar penulisan skripsi.
> - Seluruh angka diambil langsung dari output eksperimen notebook tanpa pembulatan tambahan.
