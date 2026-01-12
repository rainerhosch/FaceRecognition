"""
Face Recognition System - Main Application

A Streamlit-based face recognition system similar to FaceID.
Features:
- Face registration with multi-angle capture
- Real-time face recognition for attendance
- Face data management
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .feature-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
    }
    .feature-card p {
        margin: 0;
        opacity: 0.9;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Face Recognition System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Import database to get stats
from utils.database import get_registered_count, get_attendance_log

# Statistics
col1, col2, col3 = st.columns(3)

with col1:
    registered_count = get_registered_count()
    st.metric(
        label="Wajah Terdaftar",
        value=registered_count,
        delta=None
    )

with col2:
    attendance_log = get_attendance_log()
    today_count = len([a for a in attendance_log if a["timestamp"].split("T")[0] == str(__import__("datetime").date.today())])
    st.metric(
        label="Presensi Hari Ini",
        value=today_count,
        delta=None
    )

with col3:
    st.metric(
        label="Total Presensi",
        value=len(attendance_log),
        delta=None
    )

st.markdown("---")

# Feature cards
st.subheader("Fitur Utama")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>Registrasi Wajah</h3>
        <p>Daftarkan wajah Anda dengan capture dari berbagai sudut untuk akurasi maksimal.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Mulai Registrasi", key="reg_btn", use_container_width=True):
        st.switch_page("pages/1_Registrasi_Wajah.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>Presensi</h3>
        <p>Lakukan presensi dengan pengenalan wajah real-time. Cepat dan akurat.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Mulai Presensi", key="att_btn", use_container_width=True):
        st.switch_page("pages/2_Presensi.py")

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>Kelola Data</h3>
        <p>Lihat, update, atau hapus data wajah yang sudah terdaftar.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Kelola Data", key="manage_btn", use_container_width=True):
        st.switch_page("pages/3_Kelola_Data.py")

# Instructions
st.markdown("---")
st.subheader("Cara Penggunaan")

st.markdown("""
1. **Registrasi Wajah**
   - Buka menu "Registrasi Wajah" di sidebar
   - Masukkan nama Anda
   - Ikuti panduan untuk capture wajah dari 5 sudut berbeda
   - Klik tombol capture untuk setiap sudut
   - Simpan data wajah

2. **Presensi**
   - Buka menu "Presensi" di sidebar
   - Izinkan akses kamera
   - Hadapkan wajah ke kamera
   - Nama Anda akan muncul jika sudah terdaftar
   - Klik "Catat Presensi" untuk menyimpan

3. **Kelola Data**
   - Lihat daftar wajah yang terdaftar
   - Hapus atau update data jika diperlukan
""")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Face Recognition System - Built with Streamlit & DeepFace</p>",
    unsafe_allow_html=True
)
