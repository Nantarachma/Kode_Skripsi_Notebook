# BAB V KESIMPULAN DAN SARAN

Bab ini merangkum hasil penelitian pada Bab IV terkait optimasi hiperparameter multi-objective untuk model XGBoost pada tugas klasifikasi intrusi jaringan NF-UNSW-NB15-v3. Ringkasan difokuskan pada hasil empiris yang telah dibuktikan melalui evaluasi Pareto front, pengujian statistik, analisis kesalahan, interpretabilitas model, dan uji prototipe NIDS.

---

## 5.1 Kesimpulan

### 1. Optimasi multi-objective efektif menghasilkan solusi Pareto untuk NIDS

Dari total 90 trial (masing-masing 30 trial untuk TPE, NSGA-II, dan Random), diperoleh **15 solusi Pareto-optimal** dengan rentang Macro F1-Score **0,8565–0,8660** dan latensi inferensi **1,40–4,00 µs**. Hasil ini menunjukkan bahwa optimasi multi-objective mampu menemukan titik kompromi akurasi-kecepatan yang relevan untuk kebutuhan NIDS.

Pada evaluasi akhir data test holdout, ketiga metode menghasilkan akurasi tinggi (**0,9925–0,9927**) dan Kappa **0,9278–0,9298** (kategori *almost perfect*), sehingga pendekatan ini valid untuk membangun model yang andal.

### 2. Kualitas deteksi ketiga metode setara, tetapi TPE unggul signifikan pada efisiensi

Uji Kruskal-Wallis pada hasil 5-fold cross-validation menunjukkan:
- **F1-Score:** H = 1,8600, p = 0,3946 (tidak signifikan)
- **Inference Time:** H = 12,5000, p = 0,0019 (signifikan)

Artinya, secara statistik kualitas deteksi ketiga metode setara, namun efisiensi komputasi berbeda nyata. TPE memiliki rerata waktu inferensi CV **0,1415 detik**, lebih cepat dibanding NSGA-II (**0,2404 detik**) dan Random (**0,2473 detik**). Pada evaluasi akhir, TPE juga menjadi yang tercepat dengan latensi **2,23 µs/sample**.

Selain itu, waktu optimasi TPE juga paling singkat (**24,80 menit**) dibanding NSGA-II (**31,39 menit**) dan Random (**34,02 menit**).

### 3. Tantangan utama klasifikasi berada pada overlap antar kelas serangan

Kinerja per kelas konsisten pada ketiga metode: kelas **Normal** mencapai skor sempurna (precision/recall/F1 = 1,0000), **Malware** tinggi (recall > 0,92), **DoS** menengah (F1 sekitar 0,81–0,82), dan **Probe** terendah (F1 sekitar 0,73). 

Pola kesalahan terbesar pada confusion matrix adalah **Probe → Malware (25,8–27,3%)** dan **DoS → Malware (20,8–21,2%)**. Konsistensi pola di semua metode menunjukkan bahwa sumber error dominan berasal dari kemiripan karakteristik data antar kelas serangan, bukan semata-mata dari metode optimasi.

### 4. Interpretabilitas model menunjukkan dominasi fitur TTL

Analisis feature importance berbasis Gain mengindikasikan **`MIN_TTL`** dan **`MAX_TTL`** sebagai fitur paling dominan di semua metode, dengan rata-rata gain masing-masing **14.359,00** dan **11.645,04**. Temuan ini memperlihatkan bahwa sinyal TTL menjadi penentu utama keputusan klasifikasi.

Pada analisis pengaruh hiperparameter, `learning_rate` paling berpengaruh terhadap F1-Score, sedangkan `n_estimators` paling berpengaruh terhadap latensi pada ketiga metode.

### 5. Strategi pembobotan hybrid efektif menangani extreme imbalance

Ketidakseimbangan kelas sangat tinggi (sekitar **122,28:1**). Strategi *hybrid square-root class weighting* menurunkan rasio kontribusi efektif menjadi sekitar **11:1**, dan tetap mempertahankan kinerja tinggi pada kelas mayoritas maupun minoritas. Hasil ini menunjukkan pendekatan pembobotan biaya efektif untuk konteks NIDS dengan distribusi kelas ekstrem.

### 6. Prototipe NIDS menunjukkan kelayakan implementasi operasional

Prototipe Streamlit berhasil mengintegrasikan **7 model Pareto-optimal**. Pada skenario baseline, sistem mempertahankan prediksi normal dengan *false positive* sangat rendah. Pada skenario injection, sistem mampu menampilkan deteksi serangan secara responsif dan menjaga latensi inferensi di bawah ambang operasional dashboard. Hasil ini menegaskan bahwa model hasil optimasi layak untuk skenario *near real-time* berbasis aliran data.

---

## 5.2 Saran

Saran berikut disusun langsung berdasarkan temuan dan keterbatasan pada Bab IV.

### 1. Perkuat rekayasa fitur untuk mengurangi overlap Probe/DoS terhadap Malware

Karena kesalahan terbesar terjadi pada Probe→Malware dan DoS→Malware, penelitian lanjutan perlu menambah fitur temporal/statistik yang lebih sensitif terhadap perilaku *reconnaissance* dan serangan volumetrik agar separasi kelas meningkat.

### 2. Bandingkan dengan arsitektur model lain dan pendekatan ensemble

Untuk mengatasi batas performa pada kelas sulit, disarankan evaluasi model alternatif (misalnya deep learning berbasis urutan/waktu) dan ensemble heterogen agar kekuatan beberapa model dapat saling melengkapi.

### 3. Validasi generalisasi pada dataset dan lalu lintas jaringan yang beragam

Temuan penelitian saat ini masih pada satu dataset utama. Pengujian lintas dataset dan *live traffic* perlu dilakukan untuk memastikan bahwa pola hasil (termasuk dominasi fitur TTL dan keunggulan efisiensi TPE) benar-benar stabil pada konteks operasional berbeda.

### 4. Perluas budget optimasi dan eksplorasi metode HPO lanjutan

Jumlah trial 30 per metode sudah cukup menunjukkan pola, tetapi masih dapat ditingkatkan untuk memperkuat konvergensi Pareto front. Metode HPO lanjutan juga dapat dievaluasi untuk melihat peluang peningkatan lebih lanjut.

### 5. Kembangkan prototipe dari simulasi menuju sistem operasional

Prototipe saat ini masih berbasis data simulasi. Tahap lanjut perlu mencakup akuisisi data real-time, ekstraksi fitur flow secara online, mekanisme pembaruan model untuk menghadapi *concept drift*, serta uji beban untuk validasi skala produksi.

### 6. Tambahkan interpretabilitas lokal untuk analisis kesalahan per sampel

Analisis saat ini masih dominan global (feature importance). Untuk memahami akar salah klasifikasi secara lebih presisi, disarankan menambahkan metode interpretabilitas lokal (misalnya SHAP) agar penyebab kesalahan pada sampel individual dapat ditelusuri.

---

> **Catatan untuk pemindahan ke Word:**
> - Bab ini tidak memuat tabel atau gambar baru; seluruh data kuantitatif yang dirujuk merujuk pada tabel dan gambar yang telah disajikan pada Bab IV (Tabel 4.1–4.31, Gambar 4.1–4.25).
> - Referensi pustaka yang dirujuk dalam narasi perlu disesuaikan kembali dengan entri lengkap pada Daftar Pustaka skripsi.
> - Format angka menggunakan koma desimal (Indonesia) sesuai standar penulisan skripsi; perhatikan konversi saat pemindahan ke Word (contoh: 0,8642 bukan 0.8642).
> - Penomoran kesimpulan dan saran dapat disesuaikan dengan format penomoran yang berlaku di institusi masing-masing.
