# UTS IF A Pagi - A15 Spam Pesan

Kelompok kami membandingkan dua pendekatan **classical machine learning** untuk mengenali SMS biasa (*ham*) dan spam dari isi pesannya. Pada UTS Milestone 1 ini, kami menyiapkan data, membuat fitur TF-IDF, lalu menguji Logistic Regression sebagai baseline dan Linear SVM sebagai kandidat. Kami juga mencoba beberapa SMS sendiri untuk melihat apakah perubahan ejaan ikut mengubah prediksi.

## Identitas kelompok

| Keterangan | Isi |
|---|---|
| Project ID | A15 |
| Nama kelompok / proyek | A15 - Spam Pesan |
| Kelas | IF A Pagi |
| Mata kuliah | Machine Learning |
| Tahap | UTS Milestone 1 |
| Program studi | S-1 Teknik Informatika, Universitas Mikroskil |

| Nama anggota | NIM |
|---|---|
| Julian Caesar Eben Ezer Simorangkir | 231111575 |
| Howard Farrel Hura | 231111644 |
| Kevin Luther Kiyosaki Gultom | 231111131 |

Kode, penjelasan, dan keluaran tiap tahap bisa dilihat langsung di [notebook](UTS-Notebook.ipynb). Pembahasan lengkapnya ada di [laporan UTS](Laporan/UTS-Laporan.pdf).

## Menjalankan notebook

1. Unduh repository lewat **Code → Download ZIP**, lalu ekstrak seluruh isinya. Buka folder hasil ekstrak di VS Code. Notebook, `requirements.txt`, dan `data/` harus tetap berdampingan.
2. Siapkan **Python 3.12**, lalu buat environment dan pasang seluruh pustaka dari terminal di folder proyek. Gunakan perintah sesuai sistem operasi:

   **Windows (PowerShell atau Command Prompt):**

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   **macOS/Linux:**

   ```sh
   python3.12 -m venv .venv
   ./.venv/bin/python -m pip install -r requirements.txt
   ```

   Perintah instalasi langsung memakai Python di `.venv`, sehingga tidak bergantung pada environment yg aktif di terminal. Jika Python 3.12 belum ditemukan, pasang versi tersebut terlebih dahulu.

3. Buka `UTS-Notebook.ipynb` dengan ekstensi Python dan Jupyter di VS Code. Klik **Select Kernel** di kanan atas notebook, lalu **Select Another Kernel → Python Environments → .venv (Python 3.12)**. Kalau daftar kernel langsung menampilkan `.venv`, pilih itu. Pemilihan interpreter di terminal/status bar saja belum memastikan kernel notebook sudah sama.
4. Setelah instalasi selesai tanpa error dan kernel sudah sesuai, pilih **Restart Kernel → Run All**.

Jika memakai JupyterLab, jalankan dua perintah berikut di Windows. Untuk macOS/Linux, ganti `.\.venv\Scripts\python.exe` dengan `./.venv/bin/python`.

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab UTS-Notebook.ipynb
```

Kedua CSV sumber sudah disertakan, jadi tidak perlu mengunduh dataset lagi. Folder proyek boleh dipindahkan atau diganti namanya selama susunan di dalamnya tetap utuh. Kode memakai path relatif, berjalan di CPU, dan akan melatih ulang kedua model saat **Run All**. Instalasi pustaka memerlukan internet; proses eksperimen membaca data lokal.

Untuk mencoba pesan lain setelah training, ubah daftar `pesan_sendiri` di bagian 8 dan jalankan sel itu saja. Hasilnya masuk ke `hasil/prediksi_sendiri/`.

### Jika muncul `ModuleNotFoundError`

Contohnya, `No module named 'joblib'` berarti kernel belum bisa mengimpor modul itu. Walaupun `pip install joblib` berhasil di terminal, paketnya bisa terpasang pada Python yg berbeda. `joblib` sudah tercantum di `requirements.txt` bersama pustaka lain yg dibutuhkan.

Setelah memilih kernel `.venv` dengan Python 3.12, tambahkan **sel kode sementara di notebook** dan jalankan:

```python
import sys
print(sys.executable)
print(sys.version.split()[0])
%pip install -r requirements.txt
```

Path yg tercetak seharusnya mengarah ke `.venv` proyek. [`%pip`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip) memasang paket ke Python milik kernel tersebut. Tunggu sampai instalasi berhasil, hapus sel sementara, lalu **Restart Kernel → Run All**. Jika instalasi gagal, periksa pesan error instalasinya dulu; mengulang Run All belum akan memperbaiki paket yg belum terpasang.

## Dataset dan pembagian data

Kami memakai **Super SMS Dataset** dari Salman, Ikram, dan Kaafar (2024), berisi 67.010 baris mentah dengan label `0 = ham` dan `1 = spam`.

- **Sumber dataset:** repository peneliti [smspamresearch/spstudy](https://github.com/smspamresearch/spstudy).
- **CSV yg dipakai:** [super_sms_dataset.csv pada commit a0889f3](https://github.com/smspamresearch/spstudy/blob/a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv).
- **Publikasi pendamping:** [Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models](https://doi.org/10.1109/ACCESS.2024.3364671), IEEE Access (2024).

Kelompok kami memilih dataset ini karena isinya langsung sesuai dengan masalah spam SMS dan sudah memiliki label ham/spam, sehingga kami bisa fokus pada pengolahan teks serta perbandingan model. Jumlah datanya memungkinkan pembagian train, validation, dan test, sementara percobaan TF-IDF dengan model linear masih bisa dijalankan di CPU. Sumber peneliti dan versi CSV-nya juga jelas, sehingga data yg sama bisa dipakai saat eksperimen diulang. Hasil kami tetap terbatas pada koleksi ini, belum mewakili semua bahasa atau SMS baru.

Fungsi kedua CSV, pemeriksaan kualitas, dan hash berkas dijelaskan di [dokumentasi dataset](data/README.md).

Setelah pembersihan awal dan pengeluaran pola yg muncul lintas bagian, tersisa **66.288 SMS**: 46.301 untuk train, 9.955 untuk validation, dan 10.032 untuk test. Kedua model memakai pembagian data serta tiga fold yg sama. **Data test belum dipakai untuk evaluasi pada UTS ini.**

## Hasil eksperimen awal

Kami memakai macro-F1 sebagai metrik utama agar kelas ham dan spam mendapat bobot yg sama. Validasi silang memakai **StratifiedGroupKFold 3-fold di dalam train**. Setelah itu, masing-masing model dilatih ulang pada seluruh train dan dievaluasi pada validation terpisah.

| Peran dan konfigurasi | Macro-F1 CV, rata-rata ± simpangan baku | Macro-F1 validation | Accuracy validation |
|---|---:|---:|---:|
| Baseline: Logistic Regression + TF-IDF kata | 0,9820 ± 0,0013 | 0,9798 | 0,9808 |
| Kandidat: Linear SVM + TF-IDF kata dan karakter | 0,9909 ± 0,0014 | 0,9897 | 0,9902 |

Angka tersebut berasal dari [hasil_validasi.csv](hasil/evaluasi/hasil_validasi.csv); rincian tiap fold ada di [cv_per_fold.csv](hasil/evaluasi/cv_per_fold.csv). Pada validation, kandidat keliru menandai 29 SMS biasa sebagai spam dan meloloskan 69 spam. Rinciannya bisa diperiksa pada [prediksi per SMS](hasil/evaluasi/prediksi_validasi.csv).

Kandidat memberi skor lebih tinggi pada percobaan ini. Namun, yg dibandingkan adalah model sekaligus representasi fiturnya, sehingga kenaikan skor belum bisa dianggap berasal dari algoritma SVM saja. Percobaan ejaan samaran juga masih berupa contoh awal, belum pengujian ketahanan secara menyeluruh.

## Isi repository

| Lokasi | Isi dan kegunaannya |
|---|---|
| `UTS-Notebook.ipynb` | Alur percobaan beserta penjelasan dan hasil eksekusi. |
| `Laporan/UTS-Laporan.pdf` | Laporan UTS Milestone 1 beserta tautan repository. |
| `requirements.txt` | Versi pustaka yg dipakai untuk menjalankan notebook. |
| `data/` | Dataset mentah, pembagian awal, dan dokumentasi sumber data. |
| `hasil/preprocessing/` | Pembagian akhir serta catatan baris dan pola yg dikeluarkan. |
| `hasil/eda/` | Grafik distribusi kelas dan panjang SMS pada train. |
| `hasil/model/` | Dua pipeline terlatih beserta konfigurasi, seed, hash dataset, dan versi pustaka. |
| `hasil/evaluasi/` | Tabel perbandingan, skor per fold, confusion matrix, classification report, dan prediksi validation. |
| `hasil/learning_curve/` | Grafik dan tabel perubahan skor saat jumlah data latih ditambah. |
| `hasil/prediksi_sendiri/` | Prediksi untuk pesan yg dimasukkan pada bagian 8. |

File model Logistic Regression berukuran sekitar **1,39 MB**, sedangkan Linear SVM **3,51 MB**. Masing-masing sudah memuat TF-IDF dan modelnya. **Run All** memperbarui keluaran di folder `hasil/`; waktu training bisa berbeda sesuai komputer yg dipakai.

## Rencana setelah UTS

1. Nantinya kami akan memperbanyak pasangan SMS asli dan ejaan samaran, lalu menghitung berapa prediksi yg berubah pada validation.
2. Kami akan mencoba normalisasi ejaan seperti `D1scount` menjadi `Discount` sambil menjaga angka yg punya arti, misalnya `40%`.
3. Kesalahan prediksi akan kami periksa per pola untuk menentukan penyesuaian fitur atau parameter melalui validasi silang di train.
4. Setelah konfigurasi akhir ditetapkan, kami akan mengevaluasinya pada test yg sudah disisihkan.

Rencana tersebut belum menjadi hasil eksperimen pada tahap UTS ini.
