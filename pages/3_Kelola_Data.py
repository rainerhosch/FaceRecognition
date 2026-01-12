"""
Data Management Page

View, update, and delete registered face data.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import list_registered_faces, delete_face_data, clear_attendance_log, get_attendance_log

st.set_page_config(
    page_title="Kelola Data",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .person-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .danger-zone {
        background: #fff5f5;
        border: 1px solid #fc8181;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Kelola Data Wajah")
st.markdown("Lihat, update, atau hapus data wajah yang sudah terdaftar.")

# Tabs
tab1, tab2 = st.tabs(["Data Wajah", "Riwayat Presensi"])

with tab1:
    # Get all registered faces
    faces = list_registered_faces()
    
    if not faces:
        st.info("Belum ada wajah yang terdaftar.")
        if st.button("Daftarkan Wajah Sekarang"):
            st.switch_page("pages/1_Registrasi_Wajah.py")
    else:
        st.success(f"Total: {len(faces)} wajah terdaftar")
        
        # Search/filter
        search = st.text_input("Cari nama:", placeholder="Ketik nama untuk filter...")
        
        filtered_faces = faces
        if search:
            filtered_faces = [f for f in faces if search.lower() in f["name"].lower()]
        
        # Display faces
        for face in filtered_faces:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {face['name']}")
            
            with col2:
                st.markdown(f"**Samples:** {face['num_samples']}")
                st.caption(f"Dibuat: {face['created_at'][:10] if face['created_at'] != 'Unknown' else 'Unknown'}")
            
            with col3:
                # Delete button with confirmation
                if st.button("Hapus", key=f"del_{face['name']}", type="secondary"):
                    st.session_state[f"confirm_delete_{face['name']}"] = True
                
                if st.session_state.get(f"confirm_delete_{face['name']}", False):
                    st.warning(f"Yakin hapus '{face['name']}'?")
                    col_y, col_n = st.columns(2)
                    with col_y:
                        if st.button("Ya", key=f"yes_{face['name']}", type="primary"):
                            if delete_face_data(face['name']):
                                st.success("Berhasil dihapus!")
                                st.session_state[f"confirm_delete_{face['name']}"] = False
                                st.rerun()
                            else:
                                st.error("Gagal menghapus.")
                    with col_n:
                        if st.button("Tidak", key=f"no_{face['name']}"):
                            st.session_state[f"confirm_delete_{face['name']}"] = False
                            st.rerun()
            
            st.markdown("---")

with tab2:
    st.subheader("Riwayat Presensi")
    
    attendance = get_attendance_log()
    
    if not attendance:
        st.info("Belum ada data presensi.")
    else:
        # Display options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"Total: {len(attendance)} entri presensi")
        
        with col2:
            if st.button("Export CSV", use_container_width=True):
                import pandas as pd
                df = pd.DataFrame(attendance)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="attendance_full.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Show as table
        import pandas as pd
        df = pd.DataFrame(attendance)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
        
        # Display - sort first, then select columns
        df_sorted = df.sort_values('timestamp', ascending=False)
        st.dataframe(
            df_sorted[['date', 'time', 'name', 'confidence']],
            use_container_width=True,
            hide_index=True
        )
        
        # Statistics
        st.subheader("Statistik")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entri", len(attendance))
        
        with col2:
            unique_names = len(set([a["name"] for a in attendance]))
            st.metric("Orang Unik", unique_names)
        
        with col3:
            if attendance:
                avg_conf = sum([a["confidence"] for a in attendance]) / len(attendance)
                st.metric("Rata-rata Confidence", f"{avg_conf:.1%}")

# Danger zone
st.markdown("---")
st.subheader("Zona Bahaya")

with st.expander("Hapus Semua Data", expanded=False):
    st.warning("Tindakan ini tidak dapat dibatalkan!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Hapus Semua Data Wajah", type="secondary", use_container_width=True):
            st.session_state.confirm_delete_all_faces = True
        
        if st.session_state.get("confirm_delete_all_faces", False):
            st.error("PERINGATAN: Semua data wajah akan dihapus!")
            confirm = st.text_input("Ketik 'HAPUS' untuk konfirmasi:")
            if confirm == "HAPUS":
                # Delete all face files
                faces = list_registered_faces()
                for face in faces:
                    delete_face_data(face["name"])
                st.success("Semua data wajah berhasil dihapus!")
                st.session_state.confirm_delete_all_faces = False
                st.rerun()
    
    with col2:
        if st.button("Hapus Riwayat Presensi", type="secondary", use_container_width=True):
            if clear_attendance_log():
                st.success("Riwayat presensi berhasil dihapus!")
                st.rerun()
            else:
                st.error("Gagal menghapus riwayat.")
