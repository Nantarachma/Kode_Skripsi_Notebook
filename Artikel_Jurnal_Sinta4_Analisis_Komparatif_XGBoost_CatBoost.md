# Analisis Komparatif XGBoost dan CatBoost untuk Sistem Deteksi Intrusi pada Dataset NF-UNSW-NB15-v3

**Nama Penulis 1**<sup>1</sup>, **Nama Penulis 2**<sup>2</sup>  
<sup>1</sup>Program Studi ..., Fakultas ..., Universitas ...  
<sup>2</sup>Program Studi ..., Fakultas ..., Universitas ...  
Email: penulis@domain.ac.id

## ABSTRAK

Penelitian ini membahas kebutuhan model *Intrusion Detection System* (IDS) yang mampu mempertahankan kinerja deteksi tinggi pada data jaringan berskala besar dan tidak seimbang. Tujuan penelitian adalah membandingkan performa dua algoritma *gradient boosting*, yaitu XGBoost dan CatBoost, pada dataset NF-UNSW-NB15-v3. Metodologi penelitian menggunakan skenario klasifikasi multikelas dengan *stratified train-test split* 80:20, *preprocessing* numerik-kategorikal berbasis imputasi median dan *native categorical handling*, serta *class weighting* untuk mengatasi *imbalance*. Evaluasi dilakukan menggunakan metrik Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, ditambah validasi 5-fold *cross-validation* berbasis weighted F1. Hasil eksperimen menunjukkan XGBoost unggul pada kualitas prediksi dengan Accuracy 0.9901; Balanced Accuracy 0.8547; Precision 0.9915; Recall 0.9901; F1 0.9903; MCC 0.9061; ROC-AUC 0.9999; serta CV F1 0.9901±0.0001. CatBoost memiliki performa deteksi sedikit di bawah XGBoost (F1 0.9865), namun lebih efisien secara komputasi dengan waktu pelatihan 34.386 detik dan inferensi 0.0017 ms/sampel. Kesimpulan penelitian menegaskan bahwa XGBoost lebih tepat dipilih ketika prioritas utama adalah kualitas deteksi, sementara CatBoost menjadi alternatif ketika kebutuhan utama adalah efisiensi waktu.

**Kata kunci**: deteksi intrusi, XGBoost, CatBoost, NF-UNSW-NB15-v3, klasifikasi multikelas

## ABSTRACT

This study addresses the need for an Intrusion Detection System (IDS) model that can maintain high detection performance on large-scale and imbalanced network traffic data. The objective is to compare two gradient boosting algorithms, XGBoost and CatBoost, using the NF-UNSW-NB15-v3 dataset. The methodology applies a multiclass classification scenario with an 80:20 stratified train-test split, numerical-categorical preprocessing based on median imputation and native categorical handling, and class weighting to address class imbalance. Evaluation uses Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, and 5-fold weighted F1 cross-validation. Experimental results indicate that XGBoost outperforms CatBoost in predictive quality with Accuracy 0.9901, Balanced Accuracy 0.8547, Precision 0.9915, Recall 0.9901, F1 0.9903, MCC 0.9061, ROC-AUC 0.9999, and CV F1 0.9901±0.0001. CatBoost shows slightly lower detection performance (F1 0.9865) but better computational efficiency, with 34.386 seconds training time and 0.0017 ms/sample inference time. The study concludes that XGBoost is preferable when detection quality is the primary objective, while CatBoost is a strong option when computational efficiency is prioritized.

**Keywords**: intrusion detection, XGBoost, CatBoost, NF-UNSW-NB15-v3, multiclass classification

## PENDAHULUAN

Ancaman keamanan jaringan terus meningkat seiring pertumbuhan volume lalu lintas data dan kompleksitas pola serangan. Pada kondisi ini, sistem deteksi intrusi berbasis *machine learning* memerlukan model yang tidak hanya akurat tetapi juga stabil terhadap ketidakseimbangan kelas dan tetap efisien secara komputasi.

Dataset NF-UNSW-NB15-v3 menyediakan skenario realistis untuk evaluasi IDS karena berukuran besar dan mencerminkan distribusi kelas yang timpang. Dalam konteks tersebut, algoritma *gradient boosting* menjadi kandidat penting karena memiliki kemampuan pemodelan nonlinier yang kuat serta dukungan komputasi yang baik.

Penelitian ini difokuskan pada komparasi XGBoost dan CatBoost menggunakan alur evaluasi yang adil, reproduksibel, dan konsisten. Kontribusi utama penelitian ini adalah: (1) evaluasi komparatif dua model pada dataset yang sama dengan skenario multikelas, (2) analisis metrik prediksi dan efisiensi komputasi secara simultan, dan (3) penyajian dasar pengambilan keputusan pemilihan model untuk implementasi IDS.

## METODE PENELITIAN

### 1. Setup dan Data Eksperimen
Eksperimen dijalankan pada lingkungan Kaggle Notebook (Python 3.12.12, Linux) dengan dukungan akselerasi GPU aktif untuk XGBoost dan CatBoost. Dataset yang digunakan adalah **NF-UNSW-NB15-v3** dengan ukuran data **2,365,424 baris dan 55 kolom** saat pemanggilan awal.

### 2. Skenario Klasifikasi dan Split Data
Penelitian menggunakan skenario **multikelas**. Setelah pembersihan minimum (termasuk penghapusan duplikasi), data dibagi dengan *stratified split* menjadi data latih (80%) dan data uji (20%) agar proporsi kelas tetap terjaga.

### 3. Preprocessing
Tahap *preprocessing* dilakukan secara konsisten untuk kedua model:
- Pemisahan fitur numerik dan kategorikal.
- Penggantian nilai `inf/-inf` menjadi `NaN`.
- Imputasi median untuk fitur numerik.
- Penanganan nilai kategorikal menggunakan strategi `MISSING/UNKNOWN` dan konversi ke tipe `category`.
- Penanganan ketidakseimbangan kelas menggunakan *balanced class weight* yang dikonversi menjadi `sample_weight` pada proses pelatihan.

### 4. Konfigurasi Model Baseline
Model baseline disusun sebagai berikut:
- **XGBoost**: `n_estimators=300`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.9`, `colsample_bytree=0.9`, `objective=multi:softprob`, `enable_categorical=True`.
- **CatBoost**: `iterations=300`, `learning_rate=0.1`, `depth=6`, `loss_function=MultiClass`, dengan dukungan `cat_features` saat *fit*.

### 5. Evaluasi dan Validasi
Evaluasi model dilakukan pada data uji menggunakan metrik:
Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, ROC-AUC, serta waktu pelatihan dan inferensi per sampel. Validasi tambahan dilakukan dengan **5-fold Stratified Cross-Validation** menggunakan weighted F1.

## HASIL DAN PEMBAHASAN

### 1. Hasil Kuantitatif Utama

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | CV F1 (mean±std) | Train Time (s) | Infer/Sample (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9901 | 0.8547 | 0.9915 | 0.9901 | 0.9903 | 0.9061 | 0.9999 | 0.9901 ± 0.0001 | 54.995 | 0.0042 |
| CatBoost | 0.9861 | 0.8361 | 0.9894 | 0.9861 | 0.9865 | 0.8677 | 0.9999 | 0.9863 ± 0.0001 | 34.386 | 0.0017 |

Berdasarkan prioritas metrik F1 lalu Recall, model terbaik adalah **XGBoost**. Hal ini menunjukkan bahwa pada konteks IDS, XGBoost memberikan kualitas deteksi yang lebih kuat secara keseluruhan dan lebih stabil terhadap ketidakseimbangan kelas (tercermin dari Balanced Accuracy dan MCC yang lebih tinggi).

### 2. Ringkasan Performa per Kelas
Hasil *classification report* menunjukkan bahwa kedua model sangat baik pada kelas mayoritas (*Benign*), namun terdapat variasi performa pada kelas serangan minoritas. Secara umum, XGBoost memberikan nilai F1 yang lebih baik pada banyak kelas penting (misalnya Backdoor, Exploits, Reconnaissance, Shellcode), sehingga berdampak langsung pada peningkatan metrik agregat.

### 3. Trade-off Performa vs Efisiensi
Walaupun XGBoost unggul pada kualitas prediksi, CatBoost lebih cepat pada dua aspek komputasi:
- **Training tercepat**: CatBoost
- **Inferensi tercepat**: CatBoost

Dengan demikian, pemilihan model dapat disesuaikan dengan kebutuhan implementasi:
- Jika fokus utama adalah **akurasi deteksi ancaman**, XGBoost lebih direkomendasikan.
- Jika fokus utama adalah **efisiensi komputasi real-time**, CatBoost dapat menjadi alternatif yang kuat.

### 4. Implikasi untuk Sistem IDS
Hasil komparasi menegaskan bahwa pemilihan model IDS tidak cukup hanya mempertimbangkan satu metrik. Kombinasi metrik klasifikasi (F1, Recall, MCC, Balanced Accuracy) dan metrik operasional (waktu pelatihan/inferensi) memberikan dasar keputusan yang lebih relevan untuk penerapan nyata.

## KESIMPULAN

Penelitian komparatif pada dataset NF-UNSW-NB15-v3 menunjukkan bahwa **XGBoost** merupakan model terbaik berdasarkan prioritas kualitas deteksi (F1/Recall), dengan F1 0.9903 dan Recall 0.9901, serta stabilitas prediksi yang lebih baik (Balanced Accuracy 0.8547; MCC 0.9061). **CatBoost** tetap kompetitif dengan F1 0.9865 dan menawarkan keunggulan efisiensi komputasi melalui waktu pelatihan serta inferensi yang lebih cepat. Untuk kebutuhan IDS berbasis kualitas deteksi, XGBoost direkomendasikan; sedangkan untuk kebutuhan dengan batasan komputasi ketat, CatBoost layak dipertimbangkan.

## UCAPAN TERIMA KASIH

Penulis mengucapkan terima kasih kepada pihak yang menyediakan lingkungan eksperimen dan dataset sehingga penelitian ini dapat dilaksanakan dengan baik.

## DAFTAR PUSTAKA

[1] Sarhan, M., and Portmann, M., “NF-UNSW-NB15-v3 Dataset,” Kaggle, 2022. Available: https://www.kaggle.com/datasets/rachmanantaibnufajar/nf-unsw-nb15-v3 (accessed: 2026-04-02).  
[2] Chen, T., and Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.  
[3] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A., “CatBoost: Unbiased Boosting with Categorical Features,” in *Advances in Neural Information Processing Systems (NeurIPS 2018)*, vol. 31, 2018, pp. 6638–6648.  
[4] Pedregosa, F., et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

**Catatan pengisian akhir sebelum submit jurnal:**  
- Lengkapi identitas penulis, afiliasi, dan email korespondensi.  
- Sesuaikan gaya sitasi dan detail bibliografi dengan panduan ILTEK terbaru.  
- Jika diperlukan oleh editor, tambahkan nomor tabel/gambar sesuai layout akhir Word.
