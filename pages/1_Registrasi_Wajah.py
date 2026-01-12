"""
Face Registration Page

Allows users to register their face by capturing from multiple angles:
- Center (front facing)
- Left (turned left)
- Right (turned right)
- Up (looking up)
- Down (looking down)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.face_engine import detect_faces, get_face_encoding, get_face_locations, draw_guide_overlay
from utils.database import save_face_data, list_registered_faces

st.set_page_config(
    page_title="Registrasi Wajah",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .capture-guide {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .capture-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
    }
    .captured {
        background: #28a745;
        color: white;
    }
    .pending {
        background: #ffc107;
        color: black;
    }
    .current {
        background: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("Registrasi Wajah")
st.markdown("Daftarkan wajah Anda untuk sistem pengenalan wajah.")

# Initialize session state
if "registration_step" not in st.session_state:
    st.session_state.registration_step = 0
if "captured_encodings" not in st.session_state:
    st.session_state.captured_encodings = []
if "capture_directions" not in st.session_state:
    st.session_state.capture_directions = ["center", "left", "right", "up", "down"]
if "captured_status" not in st.session_state:
    st.session_state.captured_status = [False, False, False, False, False]

# Registration form
st.subheader("1. Masukkan Nama")
name = st.text_input("Nama lengkap:", placeholder="Contoh: John Doe")

if name:
    # Check if name already exists
    existing_faces = list_registered_faces()
    existing_names = [f["name"].lower() for f in existing_faces]
    
    if name.lower() in existing_names:
        st.warning(f"Nama '{name}' sudah terdaftar. Data baru akan ditambahkan ke data yang sudah ada.")
    
    st.subheader("2. Capture Wajah dari 5 Sudut")
    
    # Direction labels
    direction_labels = {
        "center": "Depan",
        "left": "Kiri",
        "right": "Kanan",
        "up": "Atas",
        "down": "Bawah"
    }
    
    # Show capture status
    st.markdown("**Status Capture:**")
    status_cols = st.columns(5)
    for i, (direction, captured) in enumerate(zip(st.session_state.capture_directions, st.session_state.captured_status)):
        with status_cols[i]:
            if captured:
                st.success(f"[OK] {direction_labels[direction]}")
            elif i == st.session_state.registration_step:
                st.info(f"[>>] {direction_labels[direction]}")
            else:
                st.warning(f"[--] {direction_labels[direction]}")
    
    # Current direction
    if st.session_state.registration_step < 5:
        current_direction = st.session_state.capture_directions[st.session_state.registration_step]
        
        st.markdown(f"""
        <div class="capture-guide">
            <h3>Arahkan wajah ke: {direction_labels[current_direction].upper()}</h3>
            <p>Pastikan wajah terlihat jelas di kamera</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            camera_image = st.camera_input(
                f"Capture wajah - {direction_labels[current_direction]}",
                key=f"camera_{current_direction}_{st.session_state.registration_step}"
            )
            
            if camera_image:
                # Process the captured image
                image = Image.open(camera_image)
                frame = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect faces
                face_locations = get_face_locations(frame_bgr)
                
                if len(face_locations) == 0:
                    st.error("Tidak ada wajah terdeteksi. Silakan coba lagi.")
                elif len(face_locations) > 1:
                    st.error("Terdeteksi lebih dari satu wajah. Pastikan hanya ada satu wajah di frame.")
                else:
                    # Get face encoding
                    encoding = get_face_encoding(frame_bgr, face_locations[0])
                    
                    if encoding is not None:
                        st.success("Wajah terdeteksi!")
                        
                        # Draw face box on preview
                        top, right, bottom, left = face_locations[0]
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        st.image(frame, caption="Preview", use_container_width=True)
                        
                        if st.button("Simpan Capture Ini", type="primary", use_container_width=True):
                            st.session_state.captured_encodings.append(encoding)
                            st.session_state.captured_status[st.session_state.registration_step] = True
                            st.session_state.registration_step += 1
                            st.rerun()
                    else:
                        st.error("Gagal mendapatkan encoding wajah. Silakan coba lagi.")
        
        with col2:
            st.markdown("### Panduan")
            if current_direction == "center":
                st.info("Hadapkan wajah langsung ke kamera. Pastikan pencahayaan cukup.")
            elif current_direction == "left":
                st.info("Tolehkan kepala sekitar 30 derajat ke KIRI Anda.")
            elif current_direction == "right":
                st.info("Tolehkan kepala sekitar 30 derajat ke KANAN Anda.")
            elif current_direction == "up":
                st.info("Tengadahkan kepala sedikit ke ATAS (sekitar 15 derajat).")
            elif current_direction == "down":
                st.info("Tundukkan kepala sedikit ke BAWAH (sekitar 15 derajat).")
            
            st.markdown("---")
            st.markdown(f"**Progress:** {sum(st.session_state.captured_status)}/5")
            
            # Progress bar
            progress = sum(st.session_state.captured_status) / 5
            st.progress(progress)
    
    else:
        # All captures complete
        st.success("Semua sudut sudah di-capture!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Simpan Data Wajah", type="primary", use_container_width=True):
                if save_face_data(name, st.session_state.captured_encodings):
                    st.success(f"Data wajah '{name}' berhasil disimpan!")
                    st.balloons()
                    
                    # Reset state
                    st.session_state.registration_step = 0
                    st.session_state.captured_encodings = []
                    st.session_state.captured_status = [False, False, False, False, False]
                else:
                    st.error("Gagal menyimpan data wajah.")
        
        with col2:
            if st.button("Ulangi Registrasi", use_container_width=True):
                st.session_state.registration_step = 0
                st.session_state.captured_encodings = []
                st.session_state.captured_status = [False, False, False, False, False]
                st.rerun()

else:
    st.info("Masukkan nama Anda terlebih dahulu untuk memulai registrasi.")

# Reset button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Reset Semua", use_container_width=True):
    st.session_state.registration_step = 0
    st.session_state.captured_encodings = []
    st.session_state.captured_status = [False, False, False, False, False]
    st.rerun()
