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

Beberapa penelitian terdahulu telah mengevaluasi XGBoost untuk NIDS dengan berbagai pendekatan. Yulianton dkk. (2025) melaporkan akurasi 99,67% menggunakan XGBoost dengan penyetelan hiperparameter Bayesian pada dataset UNSW-NB15, namun analisisnya berfokus pada metrik akurasi keseluruhan tanpa menelaah dampak per kelas serangan minoritas [7]. Liu dkk. (2024) mengusulkan metode deteksi intrusi berbasis seleksi fitur dan XGBoost yang mencapai kinerja tinggi, tetapi penekanannya pada seleksi fitur, bukan optimasi hiperparameter secara komprehensif [8]. More dkk. (2024) mengevaluasi performa IDS yang disempurnakan pada dataset UNSW-NB15, namun belum mengkuantifikasi kontribusi individual setiap hiperparameter terhadap kinerja model [9]. Liu (2025) mengembangkan model RFS-XGBoost untuk NIDS dengan optimasi komprehensif, akan tetapi tidak membandingkan secara eksplisit konfigurasi default versus konfigurasi yang dioptimasi [10]. Moustafa dan Slay (2015), yang memperkenalkan dataset UNSW-NB15, menunjukkan bahwa ketidakseimbangan kelas pada dataset ini memerlukan strategi penanganan khusus agar kelas serangan minoritas dapat terdeteksi secara memadai [15].

Dari tinjauan penelitian terdahulu tersebut, dapat disimpulkan bahwa meskipun XGBoost telah terbukti efektif untuk NIDS, sebagian besar penelitian masih menggunakan konfigurasi default atau *grid search* sederhana dan belum secara komprehensif mengkuantifikasi dampak optimasi hiperparameter Bayesian TPE terhadap kinerja per kelas serangan, terutama pada kelas minoritas yang paling kritis. Selain itu, analisis sensitivitas hiperparameter untuk mengidentifikasi parameter yang paling berpengaruh masih jarang dilakukan dalam konteks NIDS. Oleh karena itu, penelitian ini bertujuan untuk:

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

*Macro F1-Score* dihitung sebagai rata-rata tidak berbobot F1-Score seluruh kelas, sehingga memberikan bobot yang sama untuk setiap kelas terlepas dari ukurannya. Metrik ini cocok untuk dataset tidak seimbang karena meminimalkan bias terhadap kelas mayoritas. *Cohen's Kappa* ($\kappa$) mengukur reliabilitas klasifikasi dengan memperhitungkan kesepakatan yang terjadi secara kebetulan, memberikan evaluasi yang lebih konservatif dibandingkan akurasi pada dataset tidak seimbang [11].

---

## III. Metodologi

### A. Alur Penelitian

Gbr. 1 menampilkan alur penelitian secara keseluruhan, mulai dari pemuatan dataset hingga evaluasi komparatif.

**Gbr. 1. Diagram Alur Penelitian**

```
Dataset NF-UNSW-NB15-v3
        ↓
Pra-pemrosesan (Mapping Kelas, Pembersihan Fitur,
Imputasi Median, Standardisasi Z-score)
        ↓
Pembagian Data Stratifikasi (64/16/20%)
        ↓
Hybrid Cost-Sensitive Weighting
        ↓
   ┌────────────┬────────────┐
   ↓                         ↓
XGBoost Default     XGBoost TPE-Optimized
                    (30 Trial, Optuna)
   └────────────┴────────────┘
        ↓
Evaluasi Komparatif (Test Set)
& Analisis Interpretabilitas
```

### B. Persiapan Data

Dataset NF-UNSW-NB15-v3 [4] dimuat dengan 2.365.424 rekaman dan 54 kolom. Sepuluh kategori serangan asli dipetakan ke empat kelas (Normal 94,60%, DoS 1,08%, Probe 0,77%, Malware 3,54%) dengan rasio ketidakseimbangan 122,28:1. Setelah pembersihan fitur, imputasi median, dan standardisasi Z-score, dataset dibagi secara stratifikasi menjadi *training* (64%), validasi (16%), dan *test holdout* (20%) dengan 49 fitur final. Strategi *hybrid cost-sensitive weighting* berbasis akar kuadrat diterapkan untuk menyusutkan rasio efektif ketidakseimbangan kelas menjadi sekitar 11:1.

### C. Konfigurasi Eksperimen

Penelitian ini membandingkan dua skenario. **Skenario 1 (XGBoost Default):** model dilatih dengan 10 parameter bawaan standar XGBoost (misalnya `n_estimators`=100, `learning_rate`=0,3, `max_depth`=6) tanpa optimasi. **Skenario 2 (XGBoost TPE-Optimized):** model dioptimasi menggunakan TPE melalui *framework* Optuna dengan *single-objective* memaksimalkan *Macro F1-Score* validasi selama 30 trial. Ruang pencarian mencakup 10 hiperparameter sebagaimana ditampilkan pada Tabel I.

**Tabel I. Ruang Pencarian Hiperparameter TPE**

| Parameter | Tipe | Rentang |
|-----------|------|---------|
| `n_estimators` | Integer | [500, 2000], step=100 |
| `learning_rate` | Float (log) | [0,01 – 0,3] |
| `max_depth` | Integer | [6, 12] |
| `min_child_weight` | Integer | [1, 7] |
| `max_delta_step` | Integer | [1, 8] |
| `gamma` | Float | [0,1 – 0,5] |
| `subsample` | Float | [0,6 – 0,95] |
| `colsample_bytree` | Float | [0,5 – 0,9] |
| `reg_alpha` | Float (log) | [1e-6 – 1,0] |
| `reg_lambda` | Float (log) | [1e-6 – 1,0] |

Model terbaik diambil melalui `study.best_trial` setelah 30 trial selesai [5], [6].

### D. Analisis Interpretabilitas dan Konvergensi

Pengaruh setiap hiperparameter terhadap F1-Score dikuantifikasi menggunakan *Random Forest Regressor* sebagai *surrogate model* yang dilatih pada 30 pasangan (konfigurasi, F1). Efisiensi pencarian Bayesian diukur melalui trajektori F1-Score validasi dan *best-so-far* kumulatif untuk mengidentifikasi pola konvergensi serta fase eksplorasi-eksploitasi TPE [5].

---

## IV. Hasil dan Pembahasan

### A. Kinerja Default vs TPE-Optimized

Perbandingan kinerja antara XGBoost default dan XGBoost TPE-Optimized pada data *test holdout* disajikan pada Tabel II.

**Tabel II. Perbandingan Kinerja Default vs TPE-Optimized (Test Set)**

| Metrik | XGBoost Default | XGBoost TPE-Optimized | Δ (Peningkatan) |
|--------|-----------------|----------------------|-----------------|
| Macro F1-Score | 0,8116 | **0,8629** | **+0,0513 (+6,33%)** |
| Accuracy | 0,9916 | **0,9926** | +0,0010 |
| Cohen's Kappa | 0,9188 | **0,9287** | +0,0099 |

Optimasi TPE meningkatkan *Macro F1-Score* sebesar 0,0513 poin (6,33%) dan *Cohen's Kappa* dari 0,9188 ke 0,9287 — keduanya tergolong kategori *"Almost Perfect Agreement"* (κ > 0,81). Konfigurasi terbaik ditemukan pada Trial #26 (F1 validasi 0,8648), dengan perubahan paling kritis berupa penurunan `learning_rate` dari 0,3 ke 0,0235 (12,8× lebih lambat) dan peningkatan `n_estimators` dari 100 ke 600, mencerminkan strategi *slow learning* yang menghasilkan generalisasi superior. Aktivasi regularisasi L1 agresif (`reg_alpha`=0,8388) dan pruning (`gamma`=0,3514) turut berkontribusi pada peningkatan kinerja.

### B. Analisis Kinerja per Kelas

Gbr. 2 memvisualisasikan perbandingan F1-Score per kelas serangan antara model default dan model TPE-Optimized.

**Gbr. 2. Perbandingan F1-Score per Kelas: Default vs TPE-Optimized**

```
F1-Score per Kelas (Test Set)
──────────────────────────────────────────────────
Kelas   │ Default       │ TPE-Opt       │ Δ
──────────────────────────────────────────────────
Normal  │ ████████ 1,0000 │ ████████ 1,0000 │  0,0000
DoS     │ ██████   0,7491 │ ███████  0,8181 │ +0,0690
Probe   │ █████    0,6198 │ ██████   0,7291 │ +0,1093
Malware │ ███████  0,8843 │ ████████ 0,9043 │ +0,0200
──────────────────────────────────────────────────
Macro   │          0,8116 │          0,8629 │ +0,0513
```

Dampak optimasi paling signifikan terjadi pada kelas minoritas: kelas Probe mengalami peningkatan F1 terbesar dari 0,6198 ke 0,7291 (+0,1093 atau +17,6%), diikuti DoS dari 0,7491 ke 0,8181 (+0,0690 atau +9,2%) dan Malware dari 0,8843 ke 0,9043 (+0,0200 atau +2,3%). Kelas Normal tetap sempurna (F1=1,0000) pada kedua model, mengkonfirmasi bahwa optimasi tidak mengorbankan kemampuan klasifikasi kelas mayoritas. Pola ini mengkonfirmasi bahwa **optimasi hiperparameter TPE memberikan dampak paling besar pada kelas serangan minoritas** yang justru paling kritis dalam konteks operasional NIDS.

Analisis *confusion matrix* mengungkap tiga pola kesalahan utama: Probe → Malware (25,8%), DoS → Malware (21,0%), dan Malware → Probe (4,9%). Profil *NetFlow* aktivitas *reconnaissance* (Probe) sering menyerupai tahap awal serangan eksploitasi (Malware) dalam ruang 49 dimensi fitur. Optimasi hiperparameter berhasil menurunkan tingkat kesalahan Probe dari sekitar 37% pada model default menjadi 25,8% pada model optimized, meskipun tumpang tindih fitur yang fundamental tidak dapat sepenuhnya dieliminasi hanya melalui penyetelan hiperparameter.

### C. Analisis Importance Hiperparameter dan Konvergensi TPE

Analisis *surrogate model* (Random Forest) mengidentifikasi `learning_rate` sebagai hiperparameter paling berpengaruh terhadap *Macro F1-Score* (importance 0,2809 atau 28,09%), diikuti `subsample` (0,1981) dan `gamma` (0,1279). Distribusi importance yang relatif tersebar mengindikasikan bahwa TPE berhasil mengeksplorasi interaksi antar hiperparameter secara menyeluruh. Penurunan `learning_rate` dari 0,3 ke 0,0235 merupakan perubahan konfigurasi paling kritis, sedangkan `subsample` di posisi kedua mengkonfirmasi pentingnya *stochastic gradient boosting* untuk mengurangi *overfitting*.

**Gbr. 3. Trajektori Konvergensi Optimasi TPE (30 Trial)**

```
Konvergensi Macro F1-Score Validasi — TPE (30 Trial)
──────────────────────────────────────────────────
•  Titik biru  = F1 setiap trial individual
── Garis merah = Best-so-far (F1 terbaik kumulatif)
-- Garis abu   = F1 Default (baseline)
──────────────────────────────────────────────────
Pola: Peningkatan cepat pada trial awal, kemudian
stabil. Trial terbaik (#26) ditemukan menjelang
akhir pencarian (F1 validasi = 0,8648).
```

Trajektori konvergensi memperlihatkan dua fase khas algoritma Bayesian: fase eksplorasi pada trial awal di mana TPE membangun model probabilistik dengan mengeksplorasi region luas, dan fase eksploitasi pada trial lanjut di mana pencarian difokuskan pada region menjanjikan berdasarkan rasio *l(x)/g(x)*. Ditemukannya trial terbaik (#26) pada fase lanjut mengkonfirmasi karakteristik *late improvement* yang umum pada optimasi Bayesian [5]. Temuan ini menunjukkan bahwa 30 trial Bayesian TPE sudah memadai untuk menghasilkan konfigurasi yang secara signifikan mengungguli parameter default, menjadikan pendekatan Bayesian lebih efisien dibandingkan pencarian acak pada ruang 10 hiperparameter [14].

---

## V. Kesimpulan

Penelitian ini telah mengkuantifikasi secara empiris pengaruh optimasi hiperparameter Bayesian berbasis TPE terhadap kinerja klasifikasi XGBoost pada dataset NF-UNSW-NB15-v3. Hasil eksperimen menunjukkan bahwa optimasi TPE secara substantif meningkatkan kinerja deteksi, dengan peningkatan *Macro F1-Score* sebesar 0,0513 poin (6,33%) dari 0,8116 menjadi 0,8629 dan *Cohen's Kappa* dari 0,9188 ke 0,9287, yang keduanya tergolong kategori *"Almost Perfect Agreement"*. Peningkatan ini dicapai hanya dengan 30 trial pencarian Bayesian, mendemonstrasikan efisiensi TPE dalam mengeksplorasi ruang 10 hiperparameter.

Dampak optimasi paling signifikan terjadi pada kelas serangan minoritas yang justru paling kritis untuk dideteksi dalam konteks operasional NIDS. Kelas Probe — yang paling sulit terdeteksi — mengalami peningkatan F1 terbesar dari 0,6198 ke 0,7291 (+17,6%), diikuti DoS dari 0,7491 ke 0,8181 (+9,2%), sementara kelas Normal tetap sempurna (F1=1,0000) pada kedua model, mengkonfirmasi bahwa optimasi tidak mengorbankan kemampuan klasifikasi kelas mayoritas.

Analisis *surrogate model* mengidentifikasi `learning_rate` sebagai hiperparameter paling berpengaruh (importance 0,2809), diikuti `subsample` (0,1981) dan `gamma` (0,1279). Penurunan `learning_rate` dari 0,3 ke 0,0235 (12,8× lebih lambat) yang dikombinasikan dengan peningkatan `n_estimators` dari 100 ke 600 merupakan perubahan konfigurasi paling kritis. Proses optimasi Bayesian TPE menunjukkan konvergensi yang efisien melalui mekanisme eksplorasi-eksploitasi yang terarah, dengan trial terbaik ditemukan pada fase lanjut pencarian, mengkonfirmasi bahwa TPE secara efektif memanfaatkan informasi trial sebelumnya untuk menyempurnakan konfigurasi.

Sebagai rekomendasi untuk penelitian mendatang, perlu dilakukan perbandingan efisiensi TPE dengan algoritma optimasi hiperparameter lain seperti *random search* dan *Bayesian Optimization* berbasis *Gaussian Process* pada dataset NIDS dengan karakteristik serupa, serta validasi generalisasi konfigurasi optimal yang ditemukan TPE pada dataset NIDS lain seperti CIC-IDS-2017 dan CSE-CIC-IDS-2018.

---

## Ucapan Terima Kasih

Penulis mengucapkan terima kasih kepada dosen pembimbing atas arahan dan bimbingan selama proses penelitian dan penulisan artikel ini. Terima kasih juga disampaikan kepada Program Studi Informatika, Fakultas Teknik atas dukungan fasilitas dan lingkungan akademik yang kondusif. Penghargaan diberikan kepada pengelola dataset NF-UNSW-NB15-v3 serta komunitas *open-source* pengembang XGBoost dan Optuna yang telah menyediakan *tools* yang memungkinkan penelitian ini terlaksana.

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
> - Tabel menggunakan penomoran Romawi (Tabel I, II) dan gambar menggunakan Gbr. 1, 2, 3.
> - **Angka TPE-Optimized** (Tabel II kolom "XGBoost TPE-Optimized") merupakan data faktual dari eksperimen skripsi (Trial #26, Test Set).
> - **Gbr. 3** (Trajektori Konvergensi TPE) dihasilkan oleh script sebagai `tpe_convergence_f1.png`.
