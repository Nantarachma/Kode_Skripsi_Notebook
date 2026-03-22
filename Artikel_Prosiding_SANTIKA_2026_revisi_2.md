# Pengaruh Optimasi Hiperparameter Bayesian TPE terhadap Kinerja Klasifikasi XGBoost pada Dataset NF-UNSW-NB15-v3

---

**[UBAH MANUAL] Nama Penulis**<sup>1</sup>, **[UBAH MANUAL] Nama Pembimbing**<sup>2</sup>

<sup>1,2</sup>Program Studi Informatika, Fakultas Teknik, **[UBAH MANUAL] Nama Universitas**  
Email: **[UBAH MANUAL] penulis@universitas.ac.id**

---

## Abstrak

Penelitian ini mengkaji pengaruh optimasi hiperparameter Bayesian berbasis *Tree-structured Parzen Estimator* (TPE) terhadap kinerja klasifikasi *Network Intrusion Detection System* (NIDS) menggunakan algoritma XGBoost pada dataset NF-UNSW-NB15-v3. Dataset tersebut memuat 2.365.424 rekaman aliran jaringan dengan ketidakseimbangan kelas yang ekstrem, yaitu rasio 122,28:1 antara kelas Normal dan kelas Probe. Penelitian membandingkan kinerja XGBoost dengan konfigurasi parameter bawaan (*default*) terhadap XGBoost yang dioptimasi menggunakan TPE melalui *framework* Optuna dengan 30 trial pencarian dan satu fungsi tujuan tunggal, yaitu memaksimalkan *Macro F1-Score*. Pra-pemrosesan meliputi pemetaan ulang 10 label asli menjadi 4 kelas (Normal, DoS, Probe, Malware), penggantian nilai tak hingga menjadi *missing value*, imputasi median untuk 195.882 nilai hilang, standardisasi Z-score, dan strategi *hybrid cost-sensitive weighting* berbasis akar kuadrat untuk menangani ketidakseimbangan kelas. Evaluasi utama dilakukan pada *test holdout* terpisah sebesar 20%. Hasil eksperimen menunjukkan bahwa optimasi TPE meningkatkan *Macro F1-Score* pada data uji sebesar 0,0062 poin, dari 0,8570 menjadi 0,8633. Akurasi meningkat dari 0,9922 menjadi 0,9927 dan *Cohen's Kappa* meningkat dari 0,9251 menjadi 0,9293. Peningkatan per kelas terutama terjadi pada DoS (+0,0137) dan Malware (+0,0061), sementara kelas Normal tetap stabil pada F1=1,0000. Analisis *surrogate model* mengidentifikasi *learning_rate* sebagai hiperparameter paling berpengaruh terhadap F1-Score (importance 0,3856), diikuti *subsample* (0,2015) dan *colsample_bytree* (0,1189). Temuan ini mengindikasikan bahwa optimasi hiperparameter Bayesian TPE dapat meningkatkan kemampuan deteksi serangan jaringan, terutama pada kelas minoritas.

**Kata Kunci:** *Network Intrusion Detection*, XGBoost, *Bayesian Optimization*, *Tree-structured Parzen Estimator*, NF-UNSW-NB15-v3, Optimasi Hiperparameter, Klasifikasi Multi-Kelas

---

## I. Pendahuluan

Ancaman keamanan siber terus berkembang seiring meningkatnya kompleksitas jaringan komputer modern. *Network Intrusion Detection System* (NIDS) berbasis *machine learning* telah menjadi komponen penting dalam infrastruktur keamanan siber karena mampu mendeteksi pola serangan secara otomatis, termasuk pola yang sulit dijelaskan melalui aturan statis [1]. Di antara berbagai algoritma *machine learning* untuk data tabular, XGBoost (*Extreme Gradient Boosting*) dikenal unggul karena efisiensi komputasi, regularisasi yang kuat, dan performa yang konsisten pada data berdimensi tinggi [2].

Walaupun demikian, kinerja XGBoost sangat dipengaruhi oleh konfigurasi hiperparameter. Parameter seperti *learning_rate*, jumlah estimator, kedalaman pohon, dan parameter regularisasi menentukan keseimbangan antara kapasitas model dan kemampuan generalisasi. Penggunaan konfigurasi bawaan yang tidak disesuaikan dengan karakteristik data sering menghasilkan performa yang belum optimal, terutama pada data dengan distribusi kelas yang timpang [3].

Dari sisi data, lini UNSW-NB15 dikembangkan untuk menyediakan benchmark deteksi intrusi yang lebih modern dibandingkan dataset generasi lama [4], [5]. Pengembangannya kemudian diteruskan ke representasi *NetFlow* agar evaluasi model lebih dekat dengan fitur yang realistis untuk deployment jaringan [6], [7]. Dataset NF-UNSW-NB15-v3 yang digunakan pada penelitian ini merupakan rilis *NetFlow* versi mutakhir yang tersedia sebagai dataset terpisah dengan DOI resmi [8]. Pada eksperimen ini, dataset memuat 2.365.424 rekaman aliran jaringan dan menunjukkan ketidakseimbangan kelas yang sangat tinggi, sehingga evaluasi berbasis akurasi saja tidak memadai untuk menilai kemampuan deteksi pada kelas minoritas.

Pendekatan optimasi hiperparameter Bayesian melalui *Tree-structured Parzen Estimator* (TPE) [9] menawarkan strategi pencarian yang lebih efisien dibandingkan *grid search* maupun *random search*. TPE membangun model probabilistik dari trial yang telah dilakukan untuk mengarahkan pencarian ke wilayah ruang hiperparameter yang lebih menjanjikan. Implementasi modern dari pendekatan ini tersedia melalui Optuna [10], yang mendukung pencarian adaptif dan antarmuka eksperimen yang fleksibel.

Sejumlah penelitian terdahulu telah memanfaatkan XGBoost dalam konteks NIDS. Dhaliwal dkk. [11] menunjukkan bahwa XGBoost efektif untuk tugas IDS. Kasongo dan Sun [12] mengeksplorasi seleksi fitur berbasis XGBoost pada UNSW-NB15. Devan dan Khare [13] menggabungkan XGBoost dan DNN untuk klasifikasi intrusi, sedangkan Binsaeed dan Hafez [14] memadukan seleksi fitur XGBoost dengan pendekatan *deep learning* pada beberapa benchmark IDS, termasuk UNSW-NB15. More dkk. [15] kembali menelaah performa IDS pada UNSW-NB15 dan menekankan pentingnya pengolahan fitur serta pemilihan model. Akan tetapi, sebagian besar studi tersebut lebih berfokus pada seleksi fitur, model hibrida, atau evaluasi klasifier secara umum, bukan pada perbandingan langsung antara XGBoost *default* dan XGBoost yang dioptimasi dengan TPE pada NF-UNSW-NB15-v3.

Berdasarkan celah tersebut, penelitian ini bertujuan untuk: (1) mengkuantifikasi dampak optimasi hiperparameter TPE terhadap *Macro F1-Score*, akurasi, dan *Cohen's Kappa* pada dataset NF-UNSW-NB15-v3; (2) menganalisis perubahan kinerja pada tiap kelas, terutama kelas serangan minoritas; (3) mengidentifikasi hiperparameter yang paling berpengaruh melalui analisis *surrogate model*; dan (4) menelaah pola konvergensi proses optimasi Bayesian TPE.

---

## II. Tinjauan Pustaka

### A. XGBoost untuk Klasifikasi Jaringan

XGBoost merupakan implementasi *gradient boosting* yang dirancang untuk efisiensi komputasi dan kemampuan regularisasi yang baik pada data tabular [2]. Model ini membangun pohon keputusan secara bertahap, di mana setiap pohon baru dilatih untuk mengoreksi kesalahan prediksi dari kumpulan pohon sebelumnya. Dalam konteks klasifikasi jaringan, karakteristik tersebut menjadikan XGBoost cocok untuk menangani relasi nonlinier antarf fitur serta distribusi data yang besar dan tidak seimbang.

Sejumlah studi menguatkan posisi XGBoost dalam NIDS. Dhaliwal dkk. [11] menunjukkan efektivitas XGBoost sebagai classifier IDS, Kasongo dan Sun [12] menunjukkan manfaat seleksi fitur berbasis XGBoost pada UNSW-NB15, Devan dan Khare [13] mengusulkan model hibrida XGBoost-DNN, Binsaeed dan Hafez [14] menelaah kombinasi XGBoost dan *deep learning* pada beberapa benchmark IDS, dan More dkk. [15] menyoroti kembali tantangan evaluasi IDS pada UNSW-NB15. Temuan-temuan ini memperlihatkan bahwa XGBoost relevan untuk NIDS, tetapi efek optimasi hiperparameter masih perlu dikuantifikasi lebih rinci pada metrik yang sensitif terhadap kelas minoritas.

### B. Tree-structured Parzen Estimator (TPE)

TPE adalah algoritma optimasi hiperparameter Bayesian yang diperkenalkan oleh Bergstra dkk. [9]. Berbeda dari pendekatan yang memodelkan fungsi objektif secara langsung, TPE memisahkan distribusi konfigurasi yang menghasilkan skor baik dan skor buruk, lalu memilih kandidat baru berdasarkan rasio peningkatan yang diharapkan. Pendekatan ini efektif untuk ruang pencarian campuran yang terdiri atas parameter diskret dan kontinu.

Optuna menyediakan implementasi TPE yang praktis melalui `TPESampler`, termasuk dukungan *define-by-run* dan pengelolaan *study* secara sederhana [10]. Pada penelitian ini, Optuna digunakan untuk melakukan optimasi *single-objective* dengan arah *maximize* terhadap *Macro F1-Score*.

### C. Dataset NF-UNSW-NB15-v3

Dataset UNSW-NB15 mula-mula diperkenalkan oleh Moustafa dan Slay sebagai benchmark NIDS modern [4], lalu dievaluasi lebih lanjut dari sisi kompleksitas dan karakteristik statistiknya [5]. Sarhan dkk. kemudian mengonversi sejumlah benchmark NIDS ke representasi *NetFlow* untuk mendukung evaluasi lintas dataset yang lebih konsisten [6], dan memperluasnya melalui usulan himpunan fitur standar berbasis *NetFlow* [7]. Dataset NF-UNSW-NB15-v3 yang digunakan pada penelitian ini merupakan salah satu rilis terkini dari lini tersebut [8].

Pada berkas CSV yang digunakan dalam eksperimen ini terdapat 55 kolom mentah. Setelah kolom metadata dan label tertentu dikeluarkan dari proses pemodelan, tersisa 49 fitur numerik yang digunakan sebagai masukan model. Sepuluh label asli kemudian dikelompokkan menjadi empat kelas: Normal (Benign), DoS (DoS dan Generic), Probe (Reconnaissance dan Analysis), serta Malware (Exploits, Fuzzers, Backdoor, Shellcode, dan Worms).

### D. Metrik Evaluasi

*Macro F1-Score* dihitung sebagai rata-rata tidak berbobot F1-Score pada seluruh kelas, sehingga setiap kelas diberi kontribusi yang sama tanpa dipengaruhi ukuran kelas. Metrik ini dipilih karena lebih representatif untuk data tidak seimbang dibandingkan akurasi, terutama pada skenario multi-kelas dengan distribusi label yang timpang [20]. *Cohen's Kappa* digunakan untuk mengukur kesepakatan prediksi dengan label aktual setelah memperhitungkan peluang kesepakatan acak. Menurut Landis dan Koch, nilai Kappa di atas 0,81 dapat diinterpretasikan sebagai *Almost Perfect Agreement* [19].

---

## III. Metodologi

### A. Alur Penelitian

Penelitian dilaksanakan melalui enam tahap utama. Pertama, dataset NF-UNSW-NB15-v3 dimuat dan label asli dipetakan menjadi empat kelas target. Kedua, dilakukan pembersihan fitur, penggantian nilai tak hingga menjadi *missing value*, imputasi median, dan standardisasi. Ketiga, data dipisahkan menjadi *train+validation* dan *test holdout* secara stratifikasi. Keempat, subset *train+validation* dibagi kembali menjadi data *train* dan *validation* untuk keperluan tuning. Kelima, dua skenario model dibangun, yaitu XGBoost *default* dan XGBoost hasil optimasi TPE. Keenam, kedua model dievaluasi pada *test holdout* yang sama, lalu dilakukan analisis per kelas, analisis importance hiperparameter, dan analisis konvergensi.

**Gbr. 1. Diagram Alur Penelitian**

> **[UBAH MANUAL - SISIPKAN GAMBAR Gbr. 1 DI SINI]**  
> Contoh sintaks markdown:  
> `![Gbr. 1. Diagram Alur Penelitian](images/gbr1_alur_penelitian.png)`  
> Hapus blok placeholder ini setelah gambar final dimasukkan.

### B. Persiapan Data

Dataset berisi 2.365.424 rekaman dan 55 kolom. Distribusi empat kelas hasil pemetaan adalah Normal 2.237.731 sampel (94,60%), DoS 25.631 sampel (1,08%), Probe 18.300 sampel (0,77%), dan Malware 83.762 sampel (3,54%). Untuk menjaga evaluasi utama tetap terpisah dari proses tuning, data terlebih dahulu dibagi secara stratifikasi menjadi subset *train+validation* sebesar 80% dan *test holdout* sebesar 20%.

Pada subset *train+validation* dan *test holdout*, lima kolom dikeluarkan dari pemodelan, yaitu `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, dan `Label`. Kolom `Attack` digunakan untuk keperluan pemetaan label dan tidak dipakai sebagai fitur. Setelah proses tersebut, jumlah fitur masukan menjadi 49. Nilai `inf` dan `-inf` diubah menjadi `NaN`, kemudian 195.882 nilai hilang pada subset *train+validation* diimputasi menggunakan median tiap fitur. Median yang sama diterapkan pada *test holdout*. Standardisasi Z-score dengan `StandardScaler` kemudian dipelajari pada subset *train+validation* dan diterapkan ke subset tersebut serta ke *test holdout*.

Sesudah prapemrosesan selesai, subset *train+validation* dibagi kembali secara stratifikasi menjadi data *train* 64% dan *validation* 16%, sehingga diperoleh 1.513.871 sampel *train*, 378.468 sampel *validation*, dan 473.085 sampel *test*. Perlu dicatat bahwa statistik imputasi dan standardisasi untuk validasi internal dipelajari dari subset *train+validation* sebelum pemisahan *train-validation*. Oleh karena itu, skor validasi pada tahap tuning diperlakukan sebagai estimasi internal, sedangkan evaluasi komparatif utama penelitian tetap didasarkan pada *test holdout* terpisah.

Untuk mengatasi ketidakseimbangan kelas, penelitian ini menerapkan *hybrid cost-sensitive weighting*. Bobot awal dihitung dengan `compute_sample_weight(class_weight="balanced")`, kemudian dihaluskan dengan transformasi akar kuadrat dan dinormalisasi agar rata-rata bobot sama dengan 1,0. Pendekatan ini dipilih karena lebih ringan secara komputasi dibandingkan *oversampling* sintetis seperti SMOTE [17], sekaligus tetap sejalan dengan prinsip pembelajaran pada data timpang [16]. Setelah transformasi, bobot rata-rata per kelas menjadi sekitar 0,7600 untuk Normal, 7,1009 untuk DoS, 8,4038 untuk Probe, dan 3,9281 untuk Malware.

### C. Konfigurasi Eksperimen

Penelitian ini membandingkan dua skenario. Skenario pertama adalah XGBoost dengan konfigurasi bawaan standar, yaitu `n_estimators`=100, `learning_rate`=0,3, `max_depth`=6, `min_child_weight`=1, `max_delta_step`=0, `gamma`=0, `subsample`=1,0, `colsample_bytree`=1,0, `reg_alpha`=0, dan `reg_lambda`=1. Skenario kedua adalah XGBoost yang dioptimasi menggunakan TPE melalui Optuna selama 30 trial dengan satu tujuan, yaitu memaksimalkan *Macro F1-Score* pada data validasi [9], [10].

**Tabel I. Ruang Pencarian Hiperparameter TPE**

| Parameter | Tipe | Rentang |
|---|---|---|
| `n_estimators` | Integer | [500, 2000], step=100 |
| `learning_rate` | Float (log) | [0,01, 0,3] |
| `max_depth` | Integer | [6, 12] |
| `min_child_weight` | Integer | [1, 7] |
| `max_delta_step` | Integer | [1, 8] |
| `gamma` | Float | [0,1, 0,5] |
| `subsample` | Float | [0,6, 0,95] |
| `colsample_bytree` | Float | [0,5, 0,9] |
| `reg_alpha` | Float (log) | [1e-6, 1,0] |
| `reg_lambda` | Float (log) | [1e-6, 1,0] |

Model terbaik diambil dari `study.best_trial` setelah seluruh trial selesai. Untuk menjaga konsistensi perbandingan, model *default* dan model hasil optimasi dilatih dan dievaluasi dalam protokol data yang sama.

### D. Analisis Interpretabilitas dan Konvergensi

Setelah optimasi selesai, pengaruh relatif tiap hiperparameter terhadap *Macro F1-Score* dianalisis menggunakan *Random Forest Regressor* sebagai *surrogate model* yang dilatih pada 30 pasangan konfigurasi dan skor validasi. Analisis ini bersifat eksploratif karena jumlah titik observasinya mengikuti jumlah trial optimasi, dan secara konseptual sejalan dengan pendekatan penilaian importance hiperparameter berbasis model pengganti [18]. Selain itu, riwayat F1 validasi pada seluruh trial digunakan untuk menyusun kurva konvergensi dan *best-so-far* kumulatif, sehingga pola eksplorasi dan eksploitasi TPE dapat diamati.

---

## IV. Hasil dan Pembahasan

### A. Kinerja Default vs TPE-Optimized

Perbandingan kinerja kedua model pada *test holdout* disajikan pada Tabel II.

**Tabel II. Perbandingan Kinerja XGBoost Default dan TPE-Optimized pada Test Set**

| Metrik | XGBoost Default | XGBoost TPE-Optimized | Δ |
|---|---:|---:|---:|
| Macro F1-Score | 0,8570 | **0,8633** | **+0,0062** |
| Accuracy | 0,9922 | **0,9927** | +0,0004 |
| Cohen's Kappa | 0,9251 | **0,9293** | +0,0042 |

Optimasi TPE meningkatkan *Macro F1-Score* sebesar 0,0062 poin atau sekitar 0,72% relatif terhadap *baseline*. Akurasi meningkat 0,0004 poin dan *Cohen's Kappa* meningkat 0,0042 poin. Nilai Kappa 0,9293 termasuk kategori *Almost Perfect Agreement* menurut Landis dan Koch [19]. Walaupun peningkatan agregatnya tidak besar, arah perubahannya konsisten pada seluruh metrik utama.

Konfigurasi optimal menunjukkan perubahan struktur pembelajaran yang jelas dibandingkan parameter *default*. Nilai `learning_rate` turun dari 0,3 menjadi 0,0143, `n_estimators` naik dari 100 menjadi 1300, `max_depth` meningkat dari 6 menjadi 10, `max_delta_step` menjadi 8, dan `gamma` aktif pada 0,3559. Kombinasi ini mencerminkan strategi *slow learning* dengan jumlah pohon lebih banyak serta regularisasi yang lebih kuat, sehingga model memperoleh generalisasi yang sedikit lebih baik tanpa menurunkan performa pada kelas mayoritas.

### B. Analisis Kinerja per Kelas

Gbr. 2 menyajikan visualisasi perbandingan F1-Score per kelas, sedangkan Tabel III merangkum nilai numeriknya agar pembaca tetap dapat melihat perbedaan kuantitatif secara presisi.

**Gbr. 2. Perbandingan F1-Score per Kelas: Default vs TPE-Optimized**

> **[UBAH MANUAL - SISIPKAN GAMBAR Gbr. 2 DI SINI]**  
> Contoh sintaks markdown:  
> `![Gbr. 2. Perbandingan F1-Score per Kelas](images/gbr2_f1_per_kelas.png)`  
> Hapus blok placeholder ini setelah gambar final dimasukkan.

**Tabel III. Perbandingan F1-Score per Kelas pada Test Set**

| Kelas | Default | TPE-Optimized | Δ |
|---|---:|---:|---:|
| Normal | 1,0000 | 1,0000 | 0,0000 |
| DoS | 0,8066 | **0,8203** | **+0,0137** |
| Probe | 0,7225 | **0,7275** | **+0,0050** |
| Malware | 0,8991 | **0,9052** | **+0,0061** |
| Macro F1 | 0,8570 | **0,8633** | **+0,0062** |

Dampak optimasi muncul secara konsisten pada seluruh kelas minoritas. Kelas DoS memperoleh peningkatan terbesar, yaitu +0,0137, diikuti Malware +0,0061 dan Probe +0,0050. Kelas Normal tetap berada pada F1=1,0000 pada kedua model. Pola ini penting karena menunjukkan bahwa peningkatan skor agregat bukan berasal dari pergeseran performa pada kelas dominan, tetapi dari perbaikan deteksi pada kelas yang lebih sulit dan lebih sedikit.

Laporan klasifikasi pada model hasil optimasi menunjukkan bahwa recall DoS meningkat menjadi 0,7599 dengan presisi 0,8913, sedangkan recall Probe mencapai 0,7167 dengan presisi 0,7387. Nilai-nilai ini masih lebih rendah dibandingkan kelas Normal dan Malware, yang menunjukkan bahwa kelas Probe dan DoS tetap menjadi sumber kesalahan utama. Hal tersebut dapat dijelaskan oleh kemiripan pola *network flow* pada aktivitas *reconnaissance*, serangan generik, dan sebagian trafik eksploitasi. Dengan kata lain, optimasi hiperparameter mampu memperbaiki batas keputusan model, tetapi tidak sepenuhnya menghilangkan tumpang tindih karakteristik fitur antar kelas serangan.

### C. Importance Hiperparameter dan Konvergensi TPE

Analisis *surrogate model* mengidentifikasi `learning_rate` sebagai hiperparameter paling berpengaruh terhadap *Macro F1-Score* dengan importance 0,3856. Posisi berikutnya ditempati `subsample` sebesar 0,2015 dan `colsample_bytree` sebesar 0,1189. Hasil ini memperlihatkan bahwa kualitas generalisasi model pada eksperimen ini lebih sensitif terhadap pengaturan laju pembelajaran dan strategi *row/column sampling* dibandingkan terhadap regularisasi L1/L2. Temuan tersebut sejalan dengan pendekatan penilaian importance hiperparameter berbasis *surrogate* [18].

Gbr. 3 memperlihatkan trajektori konvergensi optimasi TPE selama 30 trial.

**Gbr. 3. Trajektori Konvergensi Optimasi TPE (30 Trial)**

> **[UBAH MANUAL - SISIPKAN GAMBAR Gbr. 3 DI SINI]**  
> Contoh sintaks markdown:  
> `![Gbr. 3. Trajektori Konvergensi Optimasi TPE](images/gbr3_konvergensi_tpe.png)`  
> Hapus blok placeholder ini setelah gambar final dimasukkan.

Riwayat optimasi menunjukkan F1 validasi awal sebesar 0,8505, nilai terbaik 0,8662, rata-rata seluruh trial 0,8607, dan simpangan baku 0,0043. Pola ini mengindikasikan bahwa pencarian TPE bergerak dari eksplorasi awal menuju eksploitasi yang lebih terarah pada fase akhir. Perbaikan terbaik muncul pada bagian akhir proses optimasi, yang konsisten dengan mekanisme TPE dalam memanfaatkan distribusi trial sebelumnya untuk memilih konfigurasi berikutnya [9]. Dengan hanya 30 trial, TPE sudah mampu menemukan konfigurasi yang mengungguli parameter *default*.

Walaupun demikian, hasil importance dan konvergensi perlu dibaca secara proporsional. Jumlah trial yang relatif terbatas membuat interpretasi importance lebih tepat diposisikan sebagai indikasi pola dominan daripada kesimpulan kausal yang final. Di sisi lain, karena kedua skenario model dievaluasi pada protokol yang sama dan dibandingkan pada *test holdout* identik, perbandingan relatif antara model *default* dan model *optimized* tetap dapat dianggap adil.

---

## V. Kesimpulan

Penelitian ini menunjukkan bahwa optimasi hiperparameter Bayesian berbasis TPE memberikan peningkatan kinerja pada klasifikasi XGBoost untuk dataset NF-UNSW-NB15-v3. Pada *test holdout*, *Macro F1-Score* meningkat dari 0,8570 menjadi 0,8633, akurasi meningkat dari 0,9922 menjadi 0,9927, dan *Cohen's Kappa* meningkat dari 0,9251 menjadi 0,9293. Nilai Kappa akhir termasuk kategori *Almost Perfect Agreement* [19].

Peningkatan performa terjadi secara konsisten pada kelas-kelas minoritas, terutama DoS dan Malware, sementara kelas Normal tetap sempurna. Analisis *surrogate model* menunjukkan bahwa `learning_rate`, `subsample`, dan `colsample_bytree` merupakan hiperparameter yang paling berpengaruh pada eksperimen ini. Secara praktis, konfigurasi terbaik cenderung bergerak ke arah *learning_rate* yang lebih kecil, jumlah estimator yang lebih besar, dan regularisasi yang lebih aktif.

Penelitian ini juga memiliki dua keterbatasan penting. Pertama, peningkatan kinerja yang diperoleh relatif kecil sehingga perlu diuji kembali pada beberapa *random seed* atau skema *split* lain untuk memastikan kestabilannya. Kedua, pada implementasi eksperimen saat ini statistik imputasi dan standardisasi dipelajari pada subset *train+validation* sebelum pemisahan *train-validation* internal. Kondisi ini tidak memengaruhi kebersihan *test holdout* 20%, tetapi dapat membuat skor validasi tuning sedikit lebih optimistis. Oleh sebab itu, penelitian lanjutan disarankan menerapkan *pipeline train-only preprocessing*, melakukan *retraining* model final pada gabungan *train+validation*, serta membandingkan TPE dengan metode lain seperti *random search* dan *Bayesian optimization* berbasis *Gaussian Process* pada dataset NIDS lain, misalnya CIC-IDS-2017 dan CSE-CIC-IDS-2018.

---

## Ucapan Terima Kasih

Penulis mengucapkan terima kasih kepada dosen pembimbing atas arahan dan bimbingan selama proses penelitian dan penulisan artikel ini. Terima kasih juga disampaikan kepada Program Studi Informatika, Fakultas Teknik atas dukungan fasilitas dan lingkungan akademik yang kondusif. Penghargaan diberikan kepada pengelola dataset NF-UNSW-NB15-v3 serta komunitas *open-source* pengembang XGBoost dan Optuna yang telah menyediakan perangkat lunak yang memungkinkan penelitian ini terlaksana.

---

## Referensi

[1] M. Khraisat, I. Gondal, P. Vamplew, dan J. Kamruzzaman, "Survey of Intrusion Detection Systems: Techniques, Datasets and Challenges," *Cybersecurity*, vol. 2, no. 1, Art. no. 20, 2019. doi: 10.1186/s42400-019-0038-7.

[2] T. Chen dan C. Guestrin, "XGBoost: A Scalable Tree Boosting System," dalam *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, hal. 785-794. doi: 10.1145/2939672.2939785.

[3] P. Probst, A.-L. Boulesteix, dan B. Bischl, "Tunability: Importance of Hyperparameters of Machine Learning Algorithms," *Journal of Machine Learning Research*, vol. 20, no. 53, hal. 1-32, 2019.

[4] N. Moustafa dan J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems (UNSW-NB15 Network Data Set)," dalam *Proc. Military Communications and Information Systems Conference (MilCIS)*, Canberra, Australia, 2015, hal. 1-6.

[5] N. Moustafa dan J. Slay, "The Evaluation of Network Anomaly Detection Systems: Statistical Analysis of the UNSW-NB15 Data Set and the Comparison with the KDD99 Data Set," *Information Security Journal: A Global Perspective*, vol. 25, no. 1-3, hal. 18-31, 2016. doi: 10.1080/19393555.2015.1125974.

[6] M. Sarhan, S. Layeghy, N. Moustafa, dan M. Portmann, "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems," dalam *Big Data Technologies and Applications*, Cham, Switzerland: Springer, 2021, hal. 117-135. doi: 10.1007/978-3-030-72080-7_7.

[7] M. Sarhan, S. Layeghy, dan M. Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets," *Mobile Networks and Applications*, vol. 27, hal. 357-370, 2022. doi: 10.1007/s11036-021-01843-0.

[8] M. Luay, S. Layeghy, M. Sarhan, S. Hoseininoorbin, N. Moustafa, dan M. Portmann, *NF-UNSW-NB15-v3*, dataset, The University of Queensland, 2025. doi: 10.48610/6e0eda1.

[9] J. Bergstra, R. Bardenet, Y. Bengio, dan Y. Kegl, "Algorithms for Hyper-Parameter Optimization," dalam *Proc. 25th International Conference on Neural Information Processing Systems (NeurIPS)*, Granada, Spain, 2011, hal. 2546-2554.

[10] T. Akiba, S. Sano, T. Yanase, T. Ohta, dan M. Koyama, "Optuna: A Next-Generation Hyperparameter Optimization Framework," dalam *Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, Anchorage, AK, USA, 2019, hal. 2623-2631. doi: 10.1145/3292500.3330701.

[11] S. S. Dhaliwal, A.-A. Nahid, dan R. Abbas, "Effective Intrusion Detection System Using XGBoost," *Information*, vol. 9, no. 7, Art. no. 149, 2018. doi: 10.3390/info9070149.

[12] S. M. Kasongo dan Y. Sun, "Performance Analysis of Intrusion Detection Systems Using a Feature Selection Method on the UNSW-NB15 Dataset," *Journal of Big Data*, vol. 7, Art. no. 105, 2020. doi: 10.1186/s40537-020-00379-6.

[13] P. Devan dan N. Khare, "An Efficient XGBoost-DNN-Based Classification Model for Network Intrusion Detection System," *Neural Computing and Applications*, vol. 32, no. 16, hal. 12499-12514, 2020. doi: 10.1007/s00521-020-04708-x.

[14] K. A. Binsaeed dan A. M. Hafez, "Enhancing Intrusion Detection Systems with XGBoost Feature Selection and Deep Learning Approaches," *International Journal of Advanced Computer Science and Applications*, vol. 14, no. 5, 2023. doi: 10.14569/IJACSA.2023.0140584.

[15] S. More, M. Idrissi, H. Mahmoud, dan A. T. Asyhari, "Enhanced Intrusion Detection Systems Performance with UNSW-NB15 Data Analysis," *Algorithms*, vol. 17, no. 2, Art. no. 64, 2024. doi: 10.3390/a17020064.

[16] H. He dan E. A. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 9, hal. 1263-1284, 2009. doi: 10.1109/TKDE.2008.239.

[17] N. V. Chawla, K. W. Bowyer, L. O. Hall, dan W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, hal. 321-357, 2002. doi: 10.1613/jair.953.

[18] F. Hutter, H. H. Hoos, dan K. Leyton-Brown, "An Efficient Approach for Assessing Hyperparameter Importance," dalam *Proc. 31st International Conference on Machine Learning (ICML)*, Beijing, China, 2014, hal. 754-762.

[19] J. R. Landis dan G. G. Koch, "The Measurement of Observer Agreement for Categorical Data," *Biometrics*, vol. 33, no. 1, hal. 159-174, 1977. doi: 10.2307/2529310.

[20] M. Sokolova dan G. Lapalme, "A Systematic Analysis of Performance Measures for Classification Tasks," *Information Processing & Management*, vol. 45, no. 4, hal. 427-437, 2009. doi: 10.1016/j.ipm.2009.03.002.

---

## Penanda Revisi Manual (Hapus Bagian Ini Sebelum Submit Final)

1. **[UBAH MANUAL]** Ganti identitas penulis, pembimbing, universitas, dan email pada bagian awal artikel.
2. **[UBAH MANUAL]** Ganti placeholder **Gbr. 1** dengan gambar alur penelitian final.
3. **[UBAH MANUAL]** Ganti placeholder **Gbr. 2** dengan visualisasi perbandingan F1-Score per kelas.
4. **[UBAH MANUAL]** Ganti placeholder **Gbr. 3** dengan grafik konvergensi TPE.
5. **[OPSIONAL]** Jika masih ada ruang halaman, tambahkan *confusion matrix* setelah pembahasan per kelas. Jika ruang terbatas, cukup pertahankan Tabel III.
6. **[CEK ULANG]** Bila notebook sudah direrun setelah perbaikan *pipeline preprocessing*, sinkronkan lagi angka pada metodologi dan hasil.
7. **[BATAS REFERENSI]** Total referensi pada naskah ini dibatasi menjadi **20** dan disusun dari sumber yang relevan serta terverifikasi.
