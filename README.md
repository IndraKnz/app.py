# Analisa Sentimen: Layanan Akademik

Aplikasi Streamlit untuk analisis sentimen ulasan layanan akademik menggunakan **TF-IDF + Multinomial Naive Bayes**, dengan preprocessing Bahasa Indonesia menggunakan **Sastrawi**.

## Fitur

- Upload dataset CSV tanpa dataset dummy.
- Pemilihan kolom ulasan, sentimen/label, dan tanggal secara dinamis.
- Case folding, pembersihan URL/mention/hashtag/angka/tanda baca, stopword removal, dan stemming Sastrawi.
- TF-IDF configurable di `utils/model.py`.
- Multinomial Naive Bayes.
- Dashboard statistik sentimen dinamis.
- Prediksi ulasan baru + confidence.
- Tabel data hasil analisis, filter sentimen, pencarian, dan download CSV.
- Grafik bar dan donut/pie interaktif dengan Plotly.
- Analisis 9 aspek layanan akademik:
  1. Jadwal Ujian
  2. Hasil Studi
  3. Transkrip
  4. Pembayaran
  5. KTM
  6. Perpustakaan
  7. Kalender Akademik
  8. Roadmap
  9. Presensi
- Deteksi multi-aspek berbasis keyword.
- Accuracy, Precision, Recall, F1-Score.
- Confusion matrix.
- Top kata berdasarkan skor TF-IDF.
- Error handling untuk CSV kosong/rusak, label kurang, data terlalu sedikit, split gagal, dan input kosong.

## Struktur

```text
analisis_sentimen/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv              # opsional; aplikasi utama memakai upload CSV
├── models/
│   └── (model disiapkan runtime)
└── utils/
    ├── __init__.py
    ├── preprocessing.py
    ├── model.py
    └── aspect.py
```

## Instalasi

Gunakan Python modern (disarankan Python 3.10+).

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Jalankan:

```bash
streamlit run app.py
```

## Format CSV

Nama kolom tidak harus sama karena aplikasi menyediakan dropdown pemetaan kolom. Contoh:

```csv
Ulasan,Sentimen,Tanggal
"Pembayaran UKT mudah","Positif","2026-01-10"
"Hasil studi belum muncul","Negatif","2026-01-11"
"Jadwal ujian sudah jelas","Positif","2026-01-12"
"Presensi kadang bermasalah","Negatif","2026-01-13"
```

Kolom minimal yang diperlukan:
- kolom ulasan/teks
- kolom sentimen/label

Kolom tanggal bersifat opsional.

## Catatan metodologi

Train-test split memakai `test_size=0.2`, `random_state=42`, dan mencoba `stratify=y` terlebih dahulu. Untuk dataset yang terlalu kecil/imbalanced, aplikasi memiliki fallback non-stratified dan akan menampilkan pesan error jika split tetap tidak memungkinkan.

Model prediksi untuk ulasan baru adalah model yang dilatih pada data training. Evaluasi Accuracy/Precision/Recall/F1 dihitung pada data test.

Angka dashboard tidak di-hardcode dan seluruhnya dihitung dari dataset yang di-upload.

## Model

Saat proses training berhasil, objek TF-IDF vectorizer dan MultinomialNB tersedia di session aplikasi. Folder `models/` disediakan untuk pengembangan penyimpanan model menggunakan Joblib.

Jika ingin menyimpan model secara permanen, Anda dapat menambahkan:

```python
import joblib
joblib.dump(vectorizer, "models/tfidf.pkl")
joblib.dump(model, "models/naive_bayes.pkl")
```

Kode utama sengaja tidak menulis model ke disk secara otomatis agar setiap upload dataset membentuk model baru sesuai data pengguna.
