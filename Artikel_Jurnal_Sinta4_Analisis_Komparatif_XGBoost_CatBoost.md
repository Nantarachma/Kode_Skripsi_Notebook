# Analisis Komparatif XGBoost dan CatBoost untuk Sistem Deteksi Intrusi pada Dataset NF-UNSW-NB15-v3

**Nama Penulis 1**<sup>1</sup>, **Nama Penulis 2**<sup>2</sup>  
<sup>1</sup>Program Studi ..., Fakultas ..., Universitas ...  
<sup>2</sup>Program Studi ..., Fakultas ..., Universitas ...  
Email: penulis@domain.ac.id

## ABSTRAK

Penelitian ini membandingkan XGBoost dan CatBoost untuk *multiclass intrusion detection* pada dataset NF-UNSW-NB15-v3 yang sangat tidak seimbang. Eksperimen dibuat reproduktif di Kaggle Notebook (Python 3.12.12, seed 42) dengan pipeline identik untuk kedua model: deduplikasi data, *stratified split* 80:20, imputasi numerik dan kategorikal, serta *balanced class weighting*. Evaluasi meliputi metrik kualitas klasifikasi, efisiensi komputasi, validasi 5-fold *Stratified CV*, dan analisis per kelas. Berdasarkan aturan pemilihan F1→Recall dari output notebook, XGBoost menjadi model terbaik (F1 0,9903; Recall 0,9901; Accuracy 0,9901; MCC 0,9061; CV F1 0,9901±0,0001). CatBoost memiliki kualitas prediksi sedikit lebih rendah (F1 0,9865), tetapi lebih cepat untuk pelatihan dan inferensi (34,386 s vs 54,995 s; 0,0017 ms/sampel vs 0,0042 ms/sampel). Temuan ini menunjukkan XGBoost lebih cocok saat prioritas adalah kualitas deteksi, sedangkan CatBoost relevan untuk skenario latensi rendah.

**Kata kunci**: deteksi intrusi, XGBoost, CatBoost, NF-UNSW-NB15-v3, klasifikasi multikelas

## ABSTRACT

This study compares XGBoost and CatBoost for multiclass intrusion detection on the highly imbalanced NF-UNSW-NB15-v3 dataset. A reproducible Kaggle Notebook protocol (Python 3.12.12, seed 42) was applied with an identical pipeline for both models: deduplication, 80:20 stratified split, numeric/categorical imputation, and balanced class weighting. Evaluation covers classification quality, computational efficiency, 5-fold stratified weighted-F1 cross-validation, and class-level diagnostics. Under the F1→Recall ranking policy used in the notebook, XGBoost is selected as the best model (F1 0.9903, Recall 0.9901, Accuracy 0.9901, MCC 0.9061, CV F1 0.9901±0.0001). CatBoost yields slightly lower detection quality (F1 0.9865) but better speed (34.386 s vs 54.995 s training; 0.0017 ms/sample vs 0.0042 ms/sample inference). Therefore, XGBoost is preferable when detection quality is the top priority, while CatBoost is attractive for low-latency scenarios.

**Keywords**: intrusion detection, XGBoost, CatBoost, NF-UNSW-NB15-v3, multiclass classification

## 1. PENDAHULUAN

Ketidakseimbangan kelas pada data IDS sering menyebabkan model tampak baik pada accuracy tetapi lemah pada kelas minoritas. Karena itu, komparasi model perlu menilai *trade-off* kualitas deteksi dan efisiensi komputasi secara bersamaan. XGBoost dan CatBoost dipilih karena keduanya kuat untuk data tabular skala besar dengan relasi nonlinier dan fitur heterogen.

Penelitian ini menerapkan analisis berbasis output notebook komparatif untuk menjawab dua pertanyaan: (1) model mana yang paling kuat secara kualitas deteksi pada data tidak seimbang, dan (2) model mana yang paling efisien untuk skenario implementasi operasional.

## 2. METODE PENELITIAN

### 2.1 Setup Eksperimen dan Data

**Tabel 1. Ringkasan setup eksperimen dan data**

| Komponen | Nilai |
|---|---|
| Environment | Kaggle Notebook (Python 3.12.12, seed 42) |
| Device | GPU aktif untuk XGBoost dan CatBoost (fallback CPU otomatis) |
| Dataset | NF-UNSW-NB15-v3 |
| Ukuran data awal | 2.365.424 baris × 55 kolom |
| Jumlah kelas | 10 kelas |
| Rasio ketidakseimbangan (Benign/Worms) | 14.162,85x |

Tabel 1 menunjukkan eksperimen dilakukan pada skala data besar dan distribusi kelas sangat timpang, sehingga evaluasi harus menekankan metrik yang sensitif terhadap kelas minoritas.

### 2.2 Pipeline Pra-pemrosesan dan Model Baseline

Pipeline yang diterapkan sama untuk kedua model: deduplikasi data, *stratified split* 80:20, *label encoding* target, imputasi numerik (median), imputasi kategorikal (`MISSING/UNKNOWN`), dan *sample weighting* berbasis distribusi kelas.

**Tabel 2. Konfigurasi baseline XGBoost dan CatBoost**

| Aspek | XGBoost | CatBoost |
|---|---|---|
| Estimator | XGBClassifier | CatBoostClassifier |
| Iterasi | 300 | 300 |
| Learning rate | 0,1 | 0,1 |
| Max depth | 6 | 6 |
| Penanganan kategorikal | `enable_categorical=True` | `cat_features` saat fit |
| Objective/Loss | `multi:softprob` | `MultiClass` |

Tabel 2 menegaskan bahwa konfigurasi dibuat sebanding agar komparasi berfokus pada karakter model, bukan perbedaan pipeline.

### 2.3 Protokol Evaluasi

Evaluasi mengikuti tiga lapis: (1) metrik hold-out, (2) efisiensi komputasi, dan (3) validasi konsistensi melalui 5-fold *Stratified CV*. Aturan pemilihan model ditetapkan dari awal dengan prioritas **F1 lalu Recall**, sesuai kebutuhan IDS pada data tidak seimbang.

## 3. HASIL DAN PEMBAHASAN

### 3.1 Hasil Utama Komparasi

**Tabel 3. Metrik utama dan efisiensi (output notebook)**

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | MCC | ROC-AUC | CV F1 | Train (s) | Infer/sampel (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,9901 | 0,8547 | 0,9915 | 0,9901 | 0,9903 | 0,9061 | 0,9999 | 0,9901±0,0001 | 54,995 | 0,0042 |
| CatBoost | 0,9861 | 0,8361 | 0,9894 | 0,9861 | 0,9865 | 0,8677 | 0,9992 | 0,9863±0,0001 | 34,386 | 0,0017 |

Tabel 3 memperlihatkan XGBoost konsisten unggul pada metrik kualitas deteksi, sedangkan CatBoost unggul pada waktu latih dan latensi inferensi.

**Gambar 1. Perbandingan metrik utama model (Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC).**

Gambar 1 menegaskan dominasi XGBoost pada metrik klasifikasi inti dengan selisih kecil namun konsisten.

### 3.2 Analisis Per Kelas dan Confusion Matrix

**Tabel 4. Ringkasan indikator per kelas (output *classification report*)**

| Indikator | XGBoost | CatBoost | Interpretasi |
|---|---:|---:|---|
| Macro F1 | 0,7902 | 0,6730 | XGBoost lebih stabil antar kelas |
| Weighted F1 | 0,9903 | 0,9865 | Keduanya tinggi, XGBoost tetap unggul |
| Recall kelas Analysis | 0,9184 | 0,9918 | CatBoost lebih tinggi pada kelas ini |
| F1 kelas Worms | 0,7188 | 0,3218 | XGBoost jauh lebih konsisten |
| F1 kelas Backdoor | 0,9378 | 0,7588 | XGBoost unggul signifikan |

Tabel 4 menunjukkan CatBoost memiliki keunggulan lokal pada *recall* kelas Analysis, tetapi XGBoost lebih seimbang pada kelas minoritas penting lainnya.

**Gambar 2. Confusion matrix mentah dan ternormalisasi untuk XGBoost dan CatBoost.**

Gambar 2 memperlihatkan pola salah-klasifikasi yang lebih terkendali pada XGBoost, terutama pada kelas minoritas yang sensitif.

### 3.3 Trade-off Kualitas vs Efisiensi

**Gambar 3. Perbandingan waktu pelatihan dan latensi inferensi per sampel.**

Gambar 3 memperjelas bahwa CatBoost lebih cepat untuk operasional: sekitar 37,47% lebih cepat saat pelatihan dan sekitar 59,52% lebih rendah pada latensi inferensi per sampel dibanding XGBoost.

Walau demikian, berdasarkan aturan pemilihan F1→Recall dari notebook, model pemenang tetap **XGBoost** karena kualitas deteksinya lebih tinggi dan lebih stabil lintas kelas.

## 4. KETERBATASAN PENELITIAN

Pertama, kedua model masih berada pada konfigurasi baseline sehingga ruang peningkatan dari tuning hiperparameter masih besar. Kedua, validasi hanya pada satu dataset sehingga generalisasi lintas lingkungan belum teruji. Ketiga, metrik operasional lanjutan seperti konsumsi memori, throughput, dan energi belum dianalisis.

## 5. KESIMPULAN DAN SARAN

Output komparatif menunjukkan **XGBoost** sebagai model terbaik berdasarkan prioritas F1→Recall, dengan keunggulan kualitas deteksi dan konsistensi antar kelas. **CatBoost** tetap layak untuk skenario dengan batasan latensi ketat karena lebih cepat pada pelatihan dan inferensi.

Saran lanjutan: lakukan tuning hiperparameter terstruktur, evaluasi lintas dataset IDS, dan tambahkan metrik operasional produksi agar rekomendasi model lebih kuat untuk implementasi nyata.

## UCAPAN TERIMA KASIH

Penulis mengucapkan terima kasih kepada penyedia dataset NF-UNSW-NB15-v3 dan platform Kaggle Notebook yang mendukung pelaksanaan eksperimen.

## DAFTAR PUSTAKA

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics, 21*(1), 6. https://doi.org/10.1186/s12864-019-6413-7

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* (Vol. 31, pp. 6638–6648).

Sarhan, M., Layeghy, S., Moustafa, N., Portmann, M., & Debie, E. (2021). NetFlow datasets for machine learning-based network intrusion detection systems. *IEEE Access, 9*, 78530–78550. https://doi.org/10.1109/ACCESS.2021.3085096
