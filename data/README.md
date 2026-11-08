# Sumber dan penggunaan dataset

Kelompok kami memakai **Super SMS Dataset** yg dipublikasikan oleh Muhammad Salman, Muhammad Ikram, dan Mohamed Ali Kaafar untuk penelitian klasifikasi spam SMS.

- Repository sumber: [smspamresearch/spstudy](https://github.com/smspamresearch/spstudy).
- Berkas sumber: [super_sms_dataset.csv pada commit a0889f3](https://github.com/smspamresearch/spstudy/blob/a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv).
- Artikel: Salman, M., Ikram, M., & Kaafar, M. A. (2024). *Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models*. IEEE Access, 12, 24306–24324. [DOI: 10.1109/ACCESS.2024.3364671](https://doi.org/10.1109/ACCESS.2024.3364671).

README peneliti meminta pengguna mencantumkan artikel tersebut. Pada versi sumber yg kami gunakan, tidak ada berkas lisensi tersendiri; kami tidak menetapkan lisensi baru untuk dataset ini.

## Dua CSV di folder ini

| Berkas | Isi | Pemakaian di notebook |
|---|---|---|
| `super_sms_dataset.csv` | Data mentah: 67.010 baris, kolom `SMSes` untuk teks dan `Labels` untuk label. | Dibaca pada bagian 2 setelah SHA-256 diperiksa. |
| `pembagian_awal.csv` | Penempatan awal 66.894 ID pesan setelah pembersihan awal, dengan kolom `id_baris`, `kelompok`, dan `split`. Ini catatan pembagian eksperimen kelompok kami. | Dibaca pada bagian 3, sebelum penanganan pola yg melintasi train, validation, dan test. |

Label `0` berarti ham atau SMS biasa, sedangkan `1` berarti spam. Data mentah memuat 40.830 ham, 26.178 spam, dan 2 baris tanpa label. Notebook membaca CSV dengan encoding `cp1252`; karakter yg gagal dibaca ditandai untuk pemeriksaan saat pembersihan.

## Dari data mentah sampai pembagian akhir

Pada pembersihan awal, teks dinormalisasi dengan Unicode NFKC dan spasinya dirapikan. Baris dengan teks kosong, label tidak valid, karakter pengganti `�`, atau teks duplikat dikeluarkan. Pemeriksaan duplikat memakai teks yg sudah dirapikan dan diseragamkan hurufnya. Notebook juga memeriksa konflik label pada teks yg sama. Tahap ini mengeluarkan 116 baris.

Untuk menjaga pemisahan pola, notebook membuat kunci `kelompok`: URL dan email diseragamkan, lalu hanya huruf a–z yg dipertahankan. Kunci ini khusus untuk pengelompokan; angka dan tanda baca pada teks yg masuk ke TF-IDF tetap dipertahankan. Sebanyak 606 pesan dikeluarkan dari train dan validation karena kuncinya juga muncul di bagian lain. Hasil akhirnya:

| Bagian | Jumlah SMS | Penggunaan |
|---|---:|---|
| Train | 46.301 | Pelatihan model dan validasi silang 3-fold. |
| Validation | 9.955 | Perbandingan baseline dan kandidat pada UTS. |
| Test | 10.032 | Disisihkan untuk evaluasi setelah konfigurasi akhir ditetapkan. |

Pembagian akhir bisa ditelusuri lewat [pembagian_data.csv](../hasil/preprocessing/pembagian_data.csv). Dua berkas lain di `hasil/preprocessing/` mencatat baris yg dikeluarkan pada pembersihan awal dan penanganan pola lintas bagian. Ketiganya merupakan keluaran preprocessing, bukan tiga dataset sumber yg berbeda.

## Keutuhan berkas

Notebook memeriksa SHA-256 `super_sms_dataset.csv` berikut sebelum membaca isinya:

```text
ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3
```

Pertahankan CSV asli saat menjalankan ulang. Menyimpan ulang melalui aplikasi spreadsheet bisa mengubah encoding atau akhir baris sehingga pemeriksaan gagal. Berkas `.gitattributes` menjaga kedua CSV sumber agar tidak mengalami konversi akhir baris oleh Git.
