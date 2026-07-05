# NLP Sentiment Analysis — Amazon Fine Food Reviews

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Dataset-363.888%20Reviews-34D399" />
</p>

<p align="center">
  Proyek NLP lengkap untuk analisis sentimen biner menggunakan 10 model Machine Learning dan Deep Learning,  
  dilengkapi dengan <strong>interactive Streamlit dashboard</strong> untuk visualisasi dan perbandingan performa model.
</p>

---

## Deskripsi Proyek

Proyek ini membandingkan performa berbagai pendekatan NLP untuk tugas **Binary Sentiment Classification** pada dataset [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

- **Task:** Klasifikasi sentimen ulasan produk → **Positif** (score 4–5) / **Negatif** (score 1–2)
- **Dataset:** 363.888 ulasan setelah preprocessing (skor 3 dihapus karena netral)
- **Metrik Utama:** **F1-Macro** (karena class imbalance: Positif 84.3% vs Negatif 15.7%)

---

## Hasil Perbandingan Model

| Rank | Model | Accuracy | F1-Macro |
|:----:|-------|:--------:|:--------:|
| 🥇 1 | **DistilBERT** | 97.39% | 0.9513 |
| 🥈 2 | Word2Vec + MLP | 94.77% | 0.8986 |
| 🥉 3 | NB + N-Gram | 90.87% | 0.8500 |
| 4 | NB + BOW | 90.17% | 0.8385 |
| 5 | GloVe 300d + MLP | — | 0.8280 |
| 6 | NB + TF-IDF | 89.11% | 0.8247 |
| 7 | GloVe 50d + MLP | 88.95% | 0.7572 |
| 8 | DT + N-Gram | 83.86% | 0.7522 |
| 9 | DT + BOW | 82.97% | 0.7419 |
| 10 | DT + TF-IDF | 81.95% | 0.7322 |

> **DistilBERT** (fine-tuned 3 epochs, ~85 menit training) mencapai performa terbaik dengan F1-Macro **0.9513**.

---

## Pipeline & Metodologi

### Preprocessing
```
1. Case Folding      → Lowercase seluruh teks
2. Cleaning          → Hapus HTML, URL, karakter khusus
3. Tokenisasi        → NLTK word tokenizer
4. Stopword Removal  → Hapus kata-kata umum
5. Lemmatisasi       → Normalisasi ke bentuk dasar (WordNet)
6. Data Augmentation → Back-translation (EN→DE→EN) via MarianMT untuk kelas minoritas
```

### Model yang Diuji

| Kategori | Model | Representasi |
|----------|-------|-------------|
| Classical ML | Decision Tree | BOW, N-Gram, TF-IDF |
| Classical ML | Naive Bayes | BOW, N-Gram, TF-IDF |
| Deep Learning | MLP | GloVe 50d, GloVe 300d, Word2Vec |
| Transformer | DistilBERT | `distilbert-base-uncased` |

### K-Fold Cross Validation (5-Fold, StratifiedKFold)

| Model | Mean F1-Macro | Std Dev |
|-------|:-------------:|:-------:|
| MLP (TF-IDF) | 0.8891 | ±0.0016 |
| Complement NB (TF-IDF) | 0.8629 | ±0.0016 |
| Decision Tree (TF-IDF) | 0.7078 | ±0.0033 |

---

## Interactive Dashboard

Dashboard Streamlit interaktif dengan 8 tab:

| Tab | Konten |
|-----|--------|
| 📂 Eksplorasi Data | Distribusi kelas, histogram skor, preprocessing steps |
| 📊 Perbandingan | Bar chart F1-Macro ranking, tabel metrik lengkap |
| 🕸️ Radar Chart | Perbandingan 4 metrik secara visual (interaktif) |
| 🔲 Confusion Matrix | Heatmap interaktif + detail TP/FP/TN/FN |
| 📈 Analisis Per Kelas | Precision & Recall per kelas (Positif vs Negatif) |
| 🎓 Riwayat Training | K-Fold CV plot, training history Deep Learning |
| 🔮 Prediksi Sentimen | Live inference dengan model NB + N-Gram |
| 📝 Kesimpulan | Summary otomatis berdasarkan data aktual |

---

## Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/<username>/nlp-sentiment-amazon.git
cd nlp-sentiment-amazon
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan File yang Diperlukan

File berikut **tidak di-commit** ke repo karena ukuran besar. Unduh atau generate sendiri:

| File | Keterangan | Cara Mendapatkan |
|------|-----------|-----------------|
| `all_metrics.json` | Sudah ada di repo | — |
| `Reviews.csv` | Dataset Amazon Fine Food Reviews | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| `glove.6B.300d.txt` | GloVe embeddings 300 dimensi | [nlp.stanford.edu](https://nlp.stanford.edu/projects/glove/) |
| `*.pkl`, `*.keras` | Model terlatih | Jalankan notebook pipeline |

### 4. Jalankan Dashboard

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka di `http://localhost:8501`

---

## Struktur Proyek

```
nlp-sentiment-amazon/
│
├── dashboard.py                   # Streamlit interactive dashboard
├── requirements.txt
├── .gitignore
├── README.md
│
├── notebooks/
│   └── AIO_Pipeline.ipynb            # Pipeline NLP lengkap
│
├── results/
│   ├── all_metrics.json              # Hasil evaluasi semua model
│   └── *_metrics.json
│
├── visualizations/
│   └── *.png                         # Grafik comparison & confusion matrix
│
├── models/
│   ├── model_nb_ngram.pkl            # Model yang di-deploy (Inference)
│   ├── vectorizer_ngram.pkl
│   └── (model & tokenizer lainnya)
│
├── data/
│   └── (Dataset CSV & GloVe - diabaikan Git)
│
└── scripts/
    └── *.py                          # Script utility & preprocessing
```

---

## Requirements

```
Python       3.10+
streamlit    >= 1.35.0
pandas       >= 2.0.0
plotly       >= 5.20.0
numpy        >= 1.26.0
joblib       >= 1.3.0
scikit-learn >= 1.4.0
tensorflow   >= 2.13.0    (untuk model .keras)
transformers >= 4.40.0    (untuk DistilBERT)
gensim       >= 4.3.0     (untuk Word2Vec)
nltk         >= 3.8.0     (untuk preprocessing)
```

---

## Temuan Utama

1. **Deep Learning > Classical ML** — Model berbasis embedding (Word2Vec, GloVe) dan Transformer secara konsisten mengungguli Classical ML dalam menangani class imbalance.

2. **DistilBERT dominan** — Mencapai F1-Macro **0.9513**, unggul **5.27%** di atas model Deep Learning terbaik berikutnya (Word2Vec + MLP).

3. **NB lebih baik dari DT** — Naive Bayes secara konsisten mengungguli Decision Tree di semua representasi teks.

4. **Dimensi Embedding berpengaruh** — GloVe 300d (F1: 0.828) mengungguli GloVe 50d (F1: 0.757), membuktikan manfaat representasi dimensi lebih tinggi.

5. **K-Fold CV stabil** — Std deviasi rendah (±0.001–0.003) membuktikan model generalisasi dengan baik dan tidak overfit.

---

## Dataset

**Amazon Fine Food Reviews**
- **Sumber:** [Kaggle / SNAP Stanford](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Total original:** 568.454 ulasan
- **Setelah filtering:** 363.888 ulasan (hapus skor 3)
- **Split:** Train 70% | Val 10% | Test 20%

---

## Author & Team

**Akhmad Zamri Ardani - 202310370311406 & Achmad Rizqy Nur - 202310370311430**  
Proyek NLP untuk keperluan akademik — Analisis Sentimen pada Dataset Amazon Fine Food Reviews

---

## License

Proyek ini dibuat untuk keperluan akademik dan edukasi.

---

<p align="center">
  Dibuat dengan ❤️ menggunakan <strong>Python · Streamlit · Plotly · TensorFlow · HuggingFace Transformers</strong>
</p>
