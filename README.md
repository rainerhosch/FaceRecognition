# Face Recognition System

Sistem pengenalan wajah berbasis web menggunakan Streamlit dan DeepFace. Mirip dengan FaceID, sistem ini memungkinkan registrasi wajah dari berbagai sudut dan pengenalan wajah untuk presensi.

## Fitur Utama

- **Registrasi Wajah** - Capture wajah dari 5 sudut berbeda (depan, kiri, kanan, atas, bawah) untuk akurasi maksimal
- **Presensi** - Pengenalan wajah dengan confidence score dan logging otomatis
- **Kelola Data** - Lihat, hapus data wajah dan export riwayat presensi ke CSV

## Requirements

- Python 3.8+
- Webcam/Kamera
- Windows/Linux/macOS

## Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd FaceRecognition
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
# Menggunakan conda
conda create -n facerecog python=3.10
conda activate facerecog

# Atau menggunakan venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Catatan untuk Windows:**
Jika `dlib` gagal install, coba:
```bash
pip install cmake
pip install dlib
```

Atau install menggunakan conda:
```bash
conda install -c conda-forge dlib
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Struktur Direktori

```
FaceRecognition/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Dokumentasi ini
├── pages/
│   ├── 1_Registrasi_Wajah.py # Halaman registrasi wajah
│   ├── 2_Presensi.py         # Halaman presensi
│   └── 3_Kelola_Data.py      # Halaman manajemen data
├── utils/
│   ├── __init__.py
│   ├── face_engine.py        # Face detection & encoding engine
│   └── database.py           # Data storage module
└── data/
    ├── faces/                # Folder penyimpanan data wajah (.pkl)
    └── attendance.csv        # Log presensi
```

## Panduan Penggunaan

### 1. Registrasi Wajah

1. Buka aplikasi dan klik menu **"Registrasi Wajah"** di sidebar
2. Masukkan nama lengkap pada kolom input
3. Ikuti panduan untuk capture wajah dari **5 sudut**:
   - **Depan** - Hadapkan wajah langsung ke kamera
   - **Kiri** - Tolehkan kepala ~30° ke kiri
   - **Kanan** - Tolehkan kepala ~30° ke kanan
   - **Atas** - Tengadahkan kepala sedikit
   - **Bawah** - Tundukkan kepala sedikit
4. Klik **"Take Photo"** untuk capture setiap sudut
5. Klik **"Simpan Capture Ini"** untuk menyimpan setiap pose
6. Setelah semua sudut tercapture, klik **"Simpan Data Wajah"**

### 2. Presensi

1. Buka menu **"Presensi"** di sidebar
2. Izinkan akses kamera jika diminta
3. Klik **"Take Photo"** untuk capture wajah
4. Sistem akan menampilkan:
   - Bounding box di sekitar wajah
   - Nama yang dikenali (jika terdaftar)
   - Confidence score
5. Klik **"Catat Presensi"** untuk menyimpan ke log

### 3. Kelola Data

1. Buka menu **"Kelola Data"** di sidebar
2. Tab **"Data Wajah"**:
   - Lihat daftar wajah terdaftar
   - Hapus data wajah jika diperlukan
3. Tab **"Riwayat Presensi"**:
   - Lihat log presensi
   - Export ke CSV

## Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Web Framework | Streamlit |
| Face Detection | DeepFace + OpenCV |
| Face Encoding | Facenet512 (512-dimensional embedding) |
| Data Storage | Pickle files + CSV |
| Comparison | Cosine Distance (threshold: 0.4) |

## Konfigurasi

### Mengubah Model Face Recognition

Edit file `utils/face_engine.py`, ubah parameter `model_name`:

```python
# Opsi model: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"
def get_face_encoding(frame, face_location=None, model_name="Facenet512"):
```

### Mengubah Threshold Pengenalan

Edit parameter `threshold` di `utils/face_engine.py`:

```python
def find_best_match(..., threshold=0.4):  # Lower = stricter
```

## Troubleshooting

### 1. Error: "No module named 'dlib'"

```bash
# Windows - install Visual Studio Build Tools dulu
pip install cmake
pip install dlib

# Atau gunakan conda
conda install -c conda-forge dlib
```

### 2. Error: "Could not open camera"

- Pastikan tidak ada aplikasi lain yang menggunakan kamera
- Coba refresh browser atau restart Streamlit

### 3. TensorFlow Warning Messages

Warning dapat diabaikan. Untuk menyembunyikan:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### 4. Wajah Tidak Terdeteksi

- Pastikan pencahayaan cukup
- Pastikan wajah terlihat jelas di frame
- Coba jarak yang berbeda dari kamera

### 5. Akurasi Rendah

- Daftarkan ulang wajah dengan pencahayaan yang lebih baik
- Pastikan capture dari berbagai sudut dilakukan dengan benar
- Pertimbangkan menggunakan model yang lebih akurat (ArcFace)

## Keamanan & Privasi

- Data wajah disimpan lokal dalam format pickle (`.pkl`)
- Tidak ada data yang dikirim ke server eksternal
- Untuk lingkungan produksi, pertimbangkan:
  - Enkripsi data wajah
  - Database yang lebih aman
  - Autentikasi pengguna

## Lisensi

GPL-3.0 License - Lihat file [LICENSE](LICENSE) untuk detail.

## Kontribusi

Pull requests welcome. Untuk perubahan besar, buka issue terlebih dahulu untuk diskusi.
