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

Preprocessing disusun konsisten untuk kedua model agar komparasi adil:
1. Pisah fitur numerik dan kategorikal.
2. Ubah `inf/-inf` menjadi `NaN`.
3. Imputasi median untuk fitur numerik.
4. Isi nilai kategorikal dengan token `MISSING/UNKNOWN` dan konversi ke tipe `category`.
5. Hitung *balanced class weight* lalu konversi ke `sample_weight`.

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

### 3.3 Trade-off Kualitas Deteksi vs Efisiensi

Meskipun XGBoost unggul pada kualitas prediksi, CatBoost lebih cepat pada tahap pelatihan dan inferensi:
- **Pelatihan tercepat**: CatBoost (34,386 s vs 54,995 s).
- **Inferensi tercepat**: CatBoost (0,0017 ms/sampel vs 0,0042 ms/sampel).

Secara praktis, hasil ini menunjukkan dua skenario pemilihan:
1. **Prioritas kualitas deteksi ancaman** (misalnya SOC dengan fokus minim *missed attack*): XGBoost lebih direkomendasikan.
2. **Prioritas latensi/efisiensi komputasi** (misalnya deployment real-time dengan resource ketat): CatBoost layak dipilih.

### 3.4 Implikasi untuk Implementasi IDS

Hasil penelitian menegaskan bahwa evaluasi IDS tidak cukup hanya berfokus pada accuracy. Untuk data *imbalanced*, metrik seperti Balanced Accuracy dan MCC perlu dijadikan komponen utama keputusan bersama F1/Recall. Kombinasi metrik performa prediksi dan metrik operasional (waktu pelatihan/inferensi) menghasilkan keputusan model yang lebih relevan untuk kebutuhan lapangan.

### 3.5 Visualisasi Hasil (Gambar 1 dst)

Bagian ini merangkum visualisasi yang dihasilkan pada notebook agar dapat langsung dipetakan ke layout jurnal. Urutan penomoran mengikuti urutan kemunculan output gambar pada cell.

**Gambar 1. Distribusi Label Dataset NF-UNSW-NB15-v3 (Cell 6)**  
Menunjukkan ketimpangan distribusi kelas yang sangat tinggi (dominasi kelas *Benign*). Visual ini digunakan untuk menegaskan tantangan *class imbalance* pada konteks IDS.

**Gambar 2. Distribusi Kelas pada Data Train dan Test setelah Stratified Split (Cell 8)**  
Memperlihatkan bahwa proporsi kelas train-test tetap konsisten setelah pembagian 80:20. Gambar ini mendukung validitas protokol evaluasi.

**Gambar 3. Heatmap Metrik Detail per Kelas antar Model (Cell 19)**  
Menyajikan perbandingan metrik per kelas (precision, recall, F1) untuk XGBoost dan CatBoost. Visual ini membantu identifikasi kelas serangan yang masih menantang.

**Gambar 4. Bar Chart Metrik Utama Antar Model (Cell 21)**  
Membandingkan metrik agregat utama (Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC) secara ringkas. Digunakan untuk menunjukkan keunggulan umum XGBoost.

**Gambar 5. Confusion Matrix (Raw Count) per Model (Cell 22, visual pertama)**  
Menampilkan jumlah prediksi benar/salah per kelas untuk masing-masing model. Visual ini menunjukkan pola salah klasifikasi absolut.

**Gambar 6. Confusion Matrix (Normalized) per Model (Cell 22, visual kedua)**  
Menyajikan proporsi kesalahan/prediksi benar per kelas dalam skala relatif. Visual ini penting untuk membandingkan kemampuan model pada kelas minoritas.

**Gambar 7. Grafik Waktu Komputasi (Training dan Inferensi) Antar Model (Cell 23)**  
Membandingkan efisiensi komputasi kedua model, menegaskan CatBoost lebih cepat pada pelatihan maupun inferensi per sampel.

**Gambar 8. Heatmap Komparasi Metrik Antar Model (Cell 25)**  
Heatmap transpos untuk memudahkan pembacaan saat jumlah model sedikit. Visual ini merangkum posisi relatif kedua model pada metrik utama.

**Catatan penempatan gambar pada naskah akhir:**  
- Sisipkan file gambar hasil ekspor notebook di dekat paragraf analisis yang relevan.  
- Gunakan caption final format jurnal: `Gambar X. Judul gambar`.  
- Tambahkan sumber internal, misalnya: `Sumber: Hasil olahan penulis dari notebook analisis-komparatif-jurnal-sinta-4 (5).ipynb`.

## 4. KETERBATASAN PENELITIAN

Beberapa keterbatasan yang perlu dicatat:
1. Eksperimen masih pada konfigurasi baseline, belum mencakup *hyperparameter tuning* ekstensif.
2. Studi hanya menggunakan satu dataset (NF-UNSW-NB15-v3), sehingga generalisasi lintas dataset belum diuji.
3. Evaluasi berfokus pada metrik klasifikasi dan waktu komputasi, belum mencakup analisis konsumsi memori/energi saat deployment.

## 5. KESIMPULAN DAN SARAN

Penelitian komparatif pada NF-UNSW-NB15-v3 menunjukkan bahwa **XGBoost** merupakan model terbaik berdasarkan prioritas F1→Recall, dengan F1 0,9903, Recall 0,9901, Balanced Accuracy 0,8547, MCC 0,9061, dan CV F1 0,9901±0,0001. **CatBoost** tetap kompetitif namun berada di bawah XGBoost pada kualitas deteksi (F1 0,9865), serta unggul pada efisiensi komputasi (pelatihan dan inferensi lebih cepat).

Untuk pengembangan penelitian selanjutnya disarankan:
1. melakukan *hyperparameter optimization* terstruktur untuk kedua model,
2. menambah uji lintas dataset IDS agar validitas eksternal meningkat, dan
3. menambahkan evaluasi aspek operasional lanjutan (memori, throughput, dan stabilitas inferensi jangka panjang).

## UCAPAN TERIMA KASIH

Penulis mengucapkan terima kasih kepada penyedia dataset NF-UNSW-NB15-v3 dan platform komputasi yang mendukung pelaksanaan eksperimen.

## DAFTAR PUSTAKA

[1] Sarhan, M., and Portmann, M., “NF-UNSW-NB15-v3 Dataset,” Kaggle, 2022. Available: https://www.kaggle.com/datasets/rachmanantaibnufajar/nf-unsw-nb15-v3 (accessed: 2026-04-02).  
[2] Chen, T., and Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.  
[3] Prokhorenkova, L., Gusev, G., Vorobev, A. V., Dorogush, A. V., and Gulin, A., “CatBoost: Unbiased Boosting with Categorical Features,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018, pp. 6638–6648.  
[4] Pedregosa, F., et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

**Catatan finalisasi sebelum submit jurnal:**  
- Lengkapi identitas penulis, afiliasi, dan email korespondensi.  
- Samakan gaya sitasi dan format tabel/gambar sesuai template jurnal tujuan (Sinta 4).  
- Tambahkan nomor dan caption final untuk seluruh tabel/gambar saat layout di dokumen akhir.
