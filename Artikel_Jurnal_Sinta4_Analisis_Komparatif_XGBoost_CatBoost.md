# Analisis Komparatif XGBoost dan CatBoost untuk Sistem Deteksi Intrusi pada Dataset NF-UNSW-NB15-v3

**Nama Penulis 1**<sup>1</sup>, **Nama Penulis 2**<sup>2</sup>  
<sup>1</sup>Program Studi ..., Fakultas ..., Universitas ...  
<sup>2</sup>Program Studi ..., Fakultas ..., Universitas ...  
Email: penulis@domain.ac.id

## ABSTRAK

Penelitian ini membahas kebutuhan model *Intrusion Detection System* (IDS) yang mampu mempertahankan kualitas deteksi tinggi pada data lalu lintas jaringan berskala besar dan tidak seimbang. Tujuan penelitian adalah membandingkan performa dua algoritma *gradient boosting* (XGBoost dan CatBoost) pada dataset NF-UNSW-NB15-v3. Eksperimen dijalankan pada skenario klasifikasi multikelas (10 kelas) dengan *stratified train-test split* 80:20, *preprocessing* numerik-kategorikal berbasis imputasi median dan *native categorical handling*, serta *class weighting* untuk mengatasi *class imbalance*. Evaluasi mencakup Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, waktu pelatihan, waktu inferensi per sampel, serta validasi 5-fold *cross-validation* berbasis weighted F1. Hasil menunjukkan XGBoost unggul pada kualitas prediksi (Accuracy 0,9901; Balanced Accuracy 0,8547; Precision 0,9915; Recall 0,9901; F1 0,9903; MCC 0,9061; ROC-AUC 0,9999; CV F1 0,9901±0,0001), sedangkan CatBoost sedikit lebih rendah pada kualitas deteksi (F1 0,9865) tetapi lebih unggul pada efisiensi komputasi (waktu pelatihan 34,386 detik; inferensi 0,0017 ms/sampel). Temuan ini menunjukkan bahwa XGBoost lebih sesuai ketika prioritas utama adalah ketepatan deteksi ancaman, sementara CatBoost lebih relevan ketika kebutuhan utama adalah latensi inferensi dan efisiensi komputasi.

**Kata kunci**: deteksi intrusi, XGBoost, CatBoost, NF-UNSW-NB15-v3, klasifikasi multikelas

## ABSTRACT

This study addresses the need for an Intrusion Detection System (IDS) model that can maintain high detection quality on large-scale and imbalanced network traffic data. The objective is to compare two gradient boosting algorithms (XGBoost and CatBoost) on the NF-UNSW-NB15-v3 dataset. The experiment uses a multiclass classification setting (10 classes) with an 80:20 stratified train-test split, numerical-categorical preprocessing via median imputation and native categorical handling, and class weighting to mitigate class imbalance. Evaluation includes Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, training time, inference time per sample, and 5-fold weighted F1 cross-validation. Results indicate that XGBoost outperforms CatBoost in predictive quality (Accuracy 0.9901, Balanced Accuracy 0.8547, Precision 0.9915, Recall 0.9901, F1 0.9903, MCC 0.9061, ROC-AUC 0.9999, CV F1 0.9901±0.0001), while CatBoost is superior in computational efficiency (34.386 seconds training, 0.0017 ms/sample inference) despite slightly lower detection quality (F1 0.9865). The findings suggest XGBoost is preferable when detection effectiveness is the top priority, whereas CatBoost is a practical option when computational efficiency and low latency are prioritized.

**Keywords**: intrusion detection, XGBoost, CatBoost, NF-UNSW-NB15-v3, multiclass classification

## 1. PENDAHULUAN

Pertumbuhan lalu lintas jaringan modern meningkatkan kompleksitas pola serangan dan menuntut sistem deteksi intrusi yang akurat, stabil pada kelas minoritas, serta efisien untuk implementasi operasional. Pada konteks ini, model *gradient boosting* menjadi kandidat kuat karena kemampuan pemodelan nonlinier, robust terhadap heterogenitas fitur, dan dukungan akselerasi komputasi.

Dataset NF-UNSW-NB15-v3 dipilih karena merepresentasikan kondisi realistis data jaringan skala besar dengan distribusi kelas yang sangat timpang. Karakteristik tersebut penting untuk menguji tidak hanya performa rata-rata model, tetapi juga kemampuan model mendeteksi kelas serangan minoritas.

Penelitian ini berfokus pada komparasi baseline XGBoost dan CatBoost menggunakan protokol evaluasi yang setara dan reproduksibel. Kontribusi penelitian meliputi: (1) evaluasi kuantitatif dua model pada skenario multikelas 10 kelas yang sama, (2) analisis gabungan metrik kualitas deteksi dan metrik efisiensi komputasi, serta (3) penyusunan dasar keputusan pemilihan model IDS sesuai kebutuhan implementasi.

## 2. METODE PENELITIAN

### 2.1 Lingkungan Eksperimen dan Reprodusibilitas

Eksperimen dijalankan pada Kaggle Notebook (Linux, Python 3.12.12) dengan *random seed* 42. Akselerasi GPU terdeteksi aktif untuk XGBoost dan CatBoost. Ringkasan lingkungan disajikan pada Tabel 1.

**Tabel 1. Ringkasan Lingkungan Eksperimen**

| Komponen | Nilai |
|---|---|
| Environment | Kaggle Notebook |
| Sistem Operasi | Linux 6.6.113+ |
| Python | 3.12.12 |
| Seed | 42 |
| XGBoost GPU | AKTIF |
| CatBoost GPU | AKTIF |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| catboost | 1.2.10 |

### 2.2 Dataset dan Audit Awal

Dataset yang digunakan adalah NF-UNSW-NB15-v3 dengan ukuran awal 2.365.424 baris dan 55 kolom. Audit awal menunjukkan 14.815 baris duplikat, total nilai hilang 63.425, dan 10 kelas pada label `Attack`. Distribusi label sangat tidak seimbang, dengan kelas *Benign* sebesar 94,60% dan kelas terkecil (*Worms*) hanya 0,01%.

**Gambar 1. Distribusi Label Dataset NF-UNSW-NB15-v3 (Cell 6)**  
Visual distribusi label menegaskan ketimpangan kelas yang sangat tinggi, ditandai dominasi kelas *Benign* dan proporsi yang sangat kecil pada beberapa kelas serangan seperti *Worms*, *Analysis*, dan *Shellcode*. Pola ini menunjukkan bahwa model berpotensi bias ke kelas mayoritas apabila evaluasi hanya berfokus pada metrik global. Oleh karena itu, gambar ini menjadi dasar metodologis untuk menekankan penggunaan metrik yang lebih peka terhadap ketidakseimbangan kelas, seperti Balanced Accuracy, Recall per kelas, F1, dan MCC, agar kualitas deteksi ancaman minoritas dapat dinilai lebih adil.

**Tabel 2. Ringkasan Audit Dataset**

| Metrik | Nilai |
|---|---:|
| Jumlah baris | 2.365.424 |
| Jumlah kolom | 55 |
| Duplikat awal | 14.815 |
| Nilai hilang total | 63.425 |
| Jumlah kelas | 10 |

**Tabel 3. Distribusi Label (Sebelum Split)**

| Kelas | Jumlah | Proporsi (%) |
|---|---:|---:|
| Benign | 2.237.731 | 94,60 |
| Exploits | 42.748 | 1,81 |
| Fuzzers | 33.816 | 1,43 |
| Generic | 19.651 | 0,83 |
| Reconnaissance | 17.074 | 0,72 |
| DoS | 5.980 | 0,25 |
| Backdoor | 4.659 | 0,20 |
| Shellcode | 2.381 | 0,10 |
| Analysis | 1.226 | 0,05 |
| Worms | 158 | 0,01 |

### 2.3 Skenario Eksperimen dan Split Data

Eksperimen ditetapkan pada skenario klasifikasi multikelas (10 kelas). Setelah penghapusan duplikasi, tersisa 2.350.609 sampel dengan 54 fitur prediktor. Data dibagi secara *stratified* menjadi 80% data latih dan 20% data uji sehingga proporsi kelas train-test tetap konsisten.

**Gambar 2. Distribusi Kelas pada Data Train dan Test setelah Stratified Split (Cell 8)**  
Visual ini menunjukkan bahwa pembagian data dengan *stratified split* berhasil mempertahankan komposisi tiap kelas secara proporsional antara data latih dan data uji. Konsistensi distribusi tersebut penting untuk menghindari pergeseran distribusi kelas antar subset yang dapat menimbulkan estimasi performa bias. Dengan demikian, perbandingan performa XGBoost dan CatBoost pada tahap evaluasi menjadi lebih valid karena kedua model diuji pada kondisi distribusi yang representatif terhadap data latihnya.

**Tabel 4. Ringkasan Split Data**

| Komponen | Nilai |
|---|---|
| Sampel setelah *drop duplicates* | 2.350.609 |
| Jumlah fitur untuk model | 54 |
| Jumlah kelas | 10 |
| Proporsi train:test | 80% : 20% |
| Ukuran train | 1.880.487 × 54 |
| Ukuran test | 470.122 × 54 |

### 2.4 Tahap Preprocessing

Preprocessing disusun konsisten untuk kedua model agar komparasi adil. Tahapan yang dilakukan mencakup pemisahan fitur numerik dan kategorikal, konversi nilai `inf/-inf` menjadi `NaN`, imputasi median pada fitur numerik, pengisian nilai kategorikal menggunakan token `MISSING/UNKNOWN` disertai konversi ke tipe `category`, serta perhitungan *balanced class weight* yang kemudian dikonversi menjadi `sample_weight`.

Pada data eksperimen, teridentifikasi 52 fitur numerik dan 2 fitur kategorikal. Strategi ini mempertahankan informasi kategorikal secara *native* tanpa *one-hot encoding*.

### 2.5 Konfigurasi Baseline Model

Konfigurasi baseline dibuat sepadan: kompleksitas model 300 pohon/iterasi dengan *learning rate* 0,1 dan kedalaman 6.

**Tabel 5. Konfigurasi Baseline XGBoost dan CatBoost**

| Model | Estimator | n_estimators/iterations | learning_rate | depth/max_depth | Device | Penanganan kategori |
|---|---|---:|---:|---:|---|---|
| XGBoost | XGBClassifier | 300 | 0,1 | 6 | cuda | `enable_categorical=True` |
| CatBoost | CatBoostClassifier | 300 | 0,1 | 6 | GPU | `cat_features` saat fit |

### 2.6 Protokol Evaluasi

Evaluasi utama dilakukan pada data uji (*hold-out test*) menggunakan metrik:
Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, waktu pelatihan, dan latensi inferensi per sampel.  
Validasi tambahan dilakukan dengan 5-fold *Stratified Cross-Validation* menggunakan weighted F1.  
Peringkat model ditetapkan dengan prioritas **F1 terlebih dahulu, lalu Recall**.

## 3. HASIL DAN PEMBAHASAN

### 3.1 Hasil Komparasi Utama

**Tabel 6. Hasil Komparasi Utama XGBoost vs CatBoost**

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | CV F1 (mean±std) | Train Time (s) | Infer/Sample (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,9901 | 0,8547 | 0,9915 | 0,9901 | 0,9903 | 0,9061 | 0,9999 | 0,9901 ± 0,0001 | 54,995 | 0,0042 |
| CatBoost | 0,9861 | 0,8361 | 0,9894 | 0,9861 | 0,9865 | 0,8677 | 0,9999 | 0,9863 ± 0,0001 | 34,386 | 0,0017 |

Berdasarkan prioritas ranking F1→Recall, model terbaik adalah **XGBoost**. Selisih performa terhadap CatBoost sebesar +0,0038 (F1), +0,0040 (Recall), +0,0186 (Balanced Accuracy), dan +0,0384 (MCC). Nilai CV F1 yang sangat kecil deviasinya (std 0,0001 pada kedua model) menunjukkan konsistensi performa antarfold.

**Gambar 3. Bar Chart Metrik Utama Antar Model (Cell 21)**  
Bar chart memperlihatkan perbandingan metrik agregat utama secara langsung sehingga perbedaan performa antarmodel dapat dibaca cepat pada satu bidang visual. Terlihat bahwa XGBoost unggul tipis namun konsisten pada metrik kualitas deteksi, khususnya F1, Recall, Balanced Accuracy, dan MCC, yang relevan untuk skenario data tidak seimbang. Gambar ini memperkuat hasil tabular bahwa keunggulan XGBoost bukan hanya pada satu metrik, tetapi pada kombinasi metrik yang merepresentasikan ketepatan klasifikasi dan stabilitas prediksi.

**Gambar 4. Heatmap Komparasi Metrik Antar Model (Cell 25)**  
Heatmap komparasi metrik menyajikan posisi relatif kedua model dalam format intensitas warna sehingga pola keunggulan dapat diidentifikasi secara cepat dan menyeluruh. Dengan tampilan terstruktur per metrik, pembaca dapat melihat bahwa XGBoost cenderung memiliki nilai lebih tinggi pada metrik kualitas deteksi, sementara CatBoost tetap kompetitif pada metrik umum tertentu. Representasi ini membantu menyederhanakan interpretasi banyak metrik sekaligus tanpa kehilangan konteks komparatif antar model.

### 3.2 Analisis Performa Per Kelas

Kedua model mencapai F1 = 1,0000 pada kelas mayoritas (*Benign*), namun perbedaan menjadi jelas pada kelas minoritas. XGBoost cenderung lebih seimbang pada precision-recall kelas serangan, terutama pada kelas Backdoor, Exploits, Shellcode, dan Worms.

**Tabel 7. Perbandingan F1 Per Kelas (XGBoost vs CatBoost)**

| Kelas | Support (test) | F1 XGBoost | F1 CatBoost | ΔF1 (XGB-CB) |
|---|---:|---:|---:|---:|
| Analysis | 245 | 0,5837 | 0,4563 | +0,1274 |
| Backdoor | 932 | 0,9378 | 0,7588 | +0,1790 |
| Benign | 444.586 | 1,0000 | 1,0000 | +0,0000 |
| DoS | 1.194 | 0,5547 | 0,4361 | +0,1186 |
| Exploits | 8.549 | 0,7959 | 0,7057 | +0,0902 |
| Fuzzers | 6.763 | 0,8513 | 0,8042 | +0,0471 |
| Generic | 3.930 | 0,9260 | 0,8911 | +0,0349 |
| Reconnaissance | 3.415 | 0,8025 | 0,7566 | +0,0459 |
| Shellcode | 476 | 0,7314 | 0,5999 | +0,1315 |
| Worms | 32 | 0,7188 | 0,3218 | +0,3970 |

Pada agregat *macro average*, XGBoost (0,7902) lebih tinggi dibanding CatBoost (0,6730), menegaskan keunggulan stabilitas antar-kelas saat data tidak seimbang. Pada *weighted average*, keduanya tinggi, namun XGBoost tetap unggul (0,9903 vs 0,9865).

**Gambar 5. Heatmap Metrik Detail per Kelas antar Model (Cell 19)**  
Heatmap metrik detail per kelas memperlihatkan variasi precision, recall, dan F1 pada masing-masing kelas serangan, sehingga kelemahan model tidak tertutup oleh skor agregat yang tinggi. Dari visual ini terlihat bahwa kelas minoritas masih menjadi area paling menantang, namun XGBoost menunjukkan kestabilan yang lebih baik dibanding CatBoost pada beberapa kelas penting seperti *Backdoor*, *Exploits*, dan *Shellcode*. Gambar ini menjadi bukti bahwa evaluasi granular per kelas diperlukan untuk menilai kesiapan model IDS dalam menghadapi distribusi serangan yang tidak merata.

**Gambar 6. Confusion Matrix (Raw Count) per Model (Cell 22, visual pertama)**  
Confusion matrix mentah menampilkan jumlah prediksi benar dan salah secara absolut untuk setiap pasangan kelas aktual-prediksi pada masing-masing model. Visual ini membantu mengidentifikasi kelas mana yang paling sering tertukar, serta menunjukkan skala kesalahan nyata yang terjadi pada data uji. Informasi absolut ini penting untuk analisis operasional karena memberikan gambaran langsung tentang volume *false positive* dan *false negative* yang mungkin berdampak pada beban investigasi di lingkungan IDS.

**Gambar 7. Confusion Matrix (Normalized) per Model (Cell 22, visual kedua)**  
Confusion matrix ternormalisasi menyajikan proporsi prediksi per kelas sehingga memungkinkan perbandingan yang adil antar kelas dengan ukuran sampel berbeda jauh. Berbeda dari matriks mentah, visual ini menonjolkan kemampuan model pada kelas minoritas karena pengaruh dominasi kelas mayoritas telah dinormalisasi. Dengan pendekatan ini, kelemahan relatif pada kelas serangan langka dapat terlihat lebih jelas dan menjadi masukan penting untuk strategi peningkatan model di tahap lanjutan.

### 3.3 Trade-off Kualitas Deteksi vs Efisiensi

Meskipun XGBoost unggul pada kualitas prediksi, CatBoost lebih cepat pada tahap pelatihan dan inferensi. Waktu pelatihan CatBoost tercatat 34,386 detik, lebih cepat dibanding XGBoost 54,995 detik, dan latensi inferensinya juga lebih rendah yaitu 0,0017 ms/sampel dibanding 0,0042 ms/sampel pada XGBoost.

Secara praktis, temuan ini menunjukkan bahwa XGBoost lebih direkomendasikan ketika prioritas utama adalah kualitas deteksi ancaman dan meminimalkan *missed attack*, sedangkan CatBoost lebih layak dipilih ketika prioritas sistem adalah latensi rendah dan efisiensi komputasi untuk deployment real-time dengan sumber daya terbatas.

**Gambar 8. Grafik Waktu Komputasi (Training dan Inferensi) Antar Model (Cell 23)**  
Grafik waktu komputasi memperlihatkan perbandingan efisiensi pelatihan dan inferensi kedua model dalam konteks implementasi nyata. Terlihat bahwa CatBoost lebih unggul pada kecepatan pelatihan maupun latensi inferensi per sampel, sedangkan XGBoost unggul pada kualitas deteksi berdasarkan metrik klasifikasi. Gambar ini menegaskan adanya *trade-off* praktis antara efektivitas deteksi dan efisiensi operasional, sehingga pemilihan model perlu disesuaikan dengan prioritas sistem, apakah berfokus pada akurasi deteksi ancaman atau kebutuhan respons real-time.

### 3.4 Implikasi untuk Implementasi IDS

Hasil penelitian menegaskan bahwa evaluasi IDS tidak cukup hanya berfokus pada accuracy. Untuk data *imbalanced*, metrik seperti Balanced Accuracy dan MCC perlu dijadikan komponen utama keputusan bersama F1/Recall. Kombinasi metrik performa prediksi dan metrik operasional (waktu pelatihan/inferensi) menghasilkan keputusan model yang lebih relevan untuk kebutuhan lapangan.

## 4. KETERBATASAN PENELITIAN

Beberapa keterbatasan yang perlu dicatat adalah bahwa eksperimen masih berada pada konfigurasi baseline dan belum mencakup *hyperparameter tuning* secara ekstensif. Selain itu, studi ini hanya menggunakan satu dataset, yaitu NF-UNSW-NB15-v3, sehingga generalisasi lintas dataset belum dapat dipastikan. Evaluasi juga masih berfokus pada metrik klasifikasi dan waktu komputasi, serta belum memasukkan analisis konsumsi memori dan energi pada skenario deployment.

## 5. KESIMPULAN DAN SARAN

Penelitian komparatif pada NF-UNSW-NB15-v3 menunjukkan bahwa **XGBoost** merupakan model terbaik berdasarkan prioritas F1→Recall, dengan F1 0,9903, Recall 0,9901, Balanced Accuracy 0,8547, MCC 0,9061, dan CV F1 0,9901±0,0001. **CatBoost** tetap kompetitif namun berada di bawah XGBoost pada kualitas deteksi (F1 0,9865), serta unggul pada efisiensi komputasi (pelatihan dan inferensi lebih cepat).

Untuk pengembangan penelitian selanjutnya, disarankan melakukan *hyperparameter optimization* yang lebih terstruktur pada kedua model, menambah uji lintas dataset IDS untuk meningkatkan validitas eksternal, serta menambahkan evaluasi aspek operasional lanjutan seperti memori, throughput, dan stabilitas inferensi jangka panjang.

## UCAPAN TERIMA KASIH

Penulis mengucapkan terima kasih kepada penyedia dataset NF-UNSW-NB15-v3 dan platform komputasi yang mendukung pelaksanaan eksperimen.

## DAFTAR PUSTAKA

[1] Sarhan, M., and Portmann, M., “NF-UNSW-NB15-v3 Dataset,” Kaggle, 2022. Available: https://www.kaggle.com/datasets/rachmanantaibnufajar/nf-unsw-nb15-v3 (accessed: 2026-04-02).  
[2] Chen, T., and Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.  
[3] Prokhorenkova, L., Gusev, G., Vorobev, A. V., Dorogush, A. V., and Gulin, A., “CatBoost: Unbiased Boosting with Categorical Features,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018, pp. 6638–6648.  
[4] Pedregosa, F., et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

**Catatan finalisasi sebelum submit jurnal:**  
Sebelum submit, lengkapi identitas penulis, afiliasi, dan email korespondensi, samakan gaya sitasi serta format tabel/gambar sesuai template jurnal tujuan (Sinta 4), dan pastikan seluruh tabel/gambar telah memiliki nomor serta caption final saat layout dokumen akhir.
