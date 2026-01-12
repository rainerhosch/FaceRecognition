"""
Attendance/Presence Page

Face recognition for attendance logging.
Uses snapshot-based camera input for stability.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.face_engine import get_face_locations, get_all_face_encodings, find_best_match, draw_face_box
from utils.database import load_all_face_data, log_attendance, get_attendance_log, get_registered_count

st.set_page_config(
    page_title="Presensi",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .recognition-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .unknown-box {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Presensi")
st.markdown("Lakukan presensi dengan pengenalan wajah.")

# Check if there are registered faces
registered_count = get_registered_count()
if registered_count == 0:
    st.warning("Belum ada wajah yang terdaftar. Silakan daftarkan wajah terlebih dahulu.")
    if st.button("Ke Halaman Registrasi"):
        st.switch_page("pages/1_Registrasi_Wajah.py")
    st.stop()

# Load face data
known_encodings, known_names = load_all_face_data()
st.info(f"{registered_count} orang terdaftar dengan {len(known_encodings)} sample wajah.")

# Session state for attendance
if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Kamera")
    
    # Camera input
    camera_image = st.camera_input("Ambil foto wajah", key="attendance_camera")
    
    if camera_image:
        # Process the captured image
        image = Image.open(camera_image)
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        with st.spinner("Memproses wajah..."):
            face_locations = get_face_locations(frame_bgr)
        
        if len(face_locations) == 0:
            st.warning("Tidak ada wajah terdeteksi. Pastikan wajah terlihat jelas di kamera.")
        else:
            # Get encodings for all detected faces
            face_encodings = get_all_face_encodings(frame_bgr, face_locations)
            
            # Process each detected face
            results = []
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                name, confidence = find_best_match(known_encodings, known_names, face_encoding)
                results.append((face_location, name, confidence))
                
                # Draw face box
                if name != "Unknown":
                    color = (0, 255, 0)  # Green for recognized
                else:
                    color = (0, 0, 255)  # Red for unknown
                
                frame = draw_face_box(frame, face_location, name, confidence, color)
            
            # Display processed image
            st.image(frame, caption="Hasil Pengenalan", use_container_width=True)
            
            # Show recognition results
            for face_location, name, confidence in results:
                if name != "Unknown":
                    st.markdown(f"""
                    <div class="recognition-box">
                        <h2>{name}</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store last recognized
                    st.session_state.last_recognized = name
                    st.session_state.last_confidence = confidence
                else:
                    st.markdown(f"""
                    <div class="unknown-box">
                        <h2>Tidak Dikenal</h2>
                        <p>Wajah ini belum terdaftar dalam sistem.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Attendance button
            if st.session_state.last_recognized and st.session_state.last_recognized != "Unknown":
                if st.button("Catat Presensi", type="primary", use_container_width=True):
                    if log_attendance(st.session_state.last_recognized, st.session_state.last_confidence):
                        st.success(f"Presensi untuk '{st.session_state.last_recognized}' berhasil dicatat!")
                    else:
                        st.error("Gagal mencatat presensi.")

with col2:
    st.subheader("Log Presensi Hari Ini")
    
    # Get today's attendance
    all_attendance = get_attendance_log()
    today = str(datetime.now().date())
    today_attendance = [a for a in all_attendance if a["timestamp"].split("T")[0] == today]
    
    if today_attendance:
        # Show in reverse order (newest first)
        for entry in reversed(today_attendance[-10:]):
            timestamp = entry["timestamp"].split("T")[1].split(".")[0]
            confidence = entry["confidence"]
            st.markdown(f"""
            **{entry['name']}**  
            Waktu: {timestamp} | Conf: {confidence:.1%}
            """)
            st.markdown("---")
    else:
        st.info("Belum ada presensi hari ini.")
    
    # Quick stats
    st.subheader("Statistik")
    
    unique_today = len(set([a["name"] for a in today_attendance]))
    st.metric("Orang Presensi Hari Ini", unique_today)
    st.metric("Total Entry Hari Ini", len(today_attendance))
    
    # Export button
    st.markdown("---")
    if st.button("Export ke CSV", use_container_width=True):
        import pandas as pd
        if all_attendance:
            df = pd.DataFrame(all_attendance)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{today}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("Belum ada data untuk di-export.")
