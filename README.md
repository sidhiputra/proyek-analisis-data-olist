# 📊 Proyek Analisis Data: E-Commerce Public Dataset (Olist) ✨

Proyek akhir kelas **Belajar Fundamental Analisis Data** (Dicoding) yang menganalisis tren transaksi bulanan, performa kategori produk, demografi wilayah pelanggan, serta segmentasi pelanggan menggunakan metode **RFM Analysis**.

## 📁 Struktur Direktori
- `dashboard/`: Berisi kode aplikasi Streamlit (`dashboard.py`) dan dataset bersih (`main_data.csv`).
- `data/`: Berisi berkas dataset mentah (*E-Commerce Public Dataset*).
- `notebook.ipynb`: Berkas Jupyter Notebook yang memuat seluruh tahapan analisis data (*Data Wrangling, EDA, Visualization, & RFM Analysis*).
- `requirements.txt`: Daftar pustaka (*library*) Python yang dibutuhkan.
- `url.txt`: Tautan menuju *dashboard* yang telah di-*deploy* di Streamlit Community Cloud.

## Setup environment

```bash
pip install -r requirements.txt
```

## Run steamlit app
```
streamlit run dashboard/dashboard.py
```

## Live Dashboard
Silakan kunjungi tautan berikut untuk melihat aplikasi yang sudah di-deploy di Streamlit Cloud:
```
https://proyek-submission-fundamental-data.streamlit.app/
```