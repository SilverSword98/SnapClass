"""
SnapClass - AI-Powered Classroom Attendance Application.
Main Streamlit entry point handling routing, session state, and UI flow.
"""

import os
import streamlit as st
import numpy as np

# Import custom modules
from src.database.db import DatabaseManager
from src.pipelines.face_pipeline import FaceRecognitionPipeline
from src.pipelines.voice_pipeline import VoiceRecognitionPipeline
from src.components.ui_components import render_qr_code, face_capture_dialog, voice_capture_dialog

# --- Page Configuration ---
st.set_page_config(page_title="SnapClass", page_icon="📸", layout="centered")

# --- Initialization ---
@st.cache_resource
def init_services():
    """Initializes and caches database and pipeline instances."""
    try:
        db = DatabaseManager()
        
        # Note: Ensure these model paths exist in your deployment environment
        face_pipe = FaceRecognitionPipeline(
            shape_predictor_path="models/shape_predictor_68_face_landmarks.dat",
            face_rec_model_path="models/dlib_face_recognition_resnet_model_v1.dat"
        )
        voice_pipe = VoiceRecognitionPipeline()
        
        return db, face_pipe, voice_pipe
    except Exception as e:
        st.error(f"System Initialization Error: {e}")
        st.stop()

db_manager, face_pipeline, voice_pipeline = init_services()

# --- Session State Management ---
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "captured_face" not in st.session_state:
    st.session_state["captured_face"] = None
if "captured_voice" not in st.session_state:
    st.session_state["captured_voice"] = None

def navigate_to(page: str) -> None:
    """Helper to switch pages."""
    st.session_state["current_page"] = page
    st.rerun()

# --- Screens ---
def render_home() -> None:
    """Renders the home screen for role selection."""
    st.title("📸 Welcome to SnapClass")
    st.write("AI-Powered Classroom Attendance System")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍🎓 I am a Student", use_container_width=True):
            navigate_to("student")
    with col2:
        if st.button("👨‍🏫 I am a Teacher", use_container_width=True):
            navigate_to("teacher")

def render_student() -> None:
    """Renders the student dashboard for marking attendance."""
    st.title("👨‍🎓 Student Dashboard")
    if st.button("⬅️ Back to Home"):
        navigate_to("home")
        
    st.divider()
    
    st.subheader("Mark Attendance")
    student_id = st.text_input("Enter Student ID (Matriculation Number)")
    session_id = st.text_input("Enter Session ID (Provided by Teacher)")
    subject_id = st.text_input("Enter Subject ID")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📸 Capture Face", use_container_width=True):
            face_capture_dialog()
            
        if st.session_state["captured_face"] is not None:
            st.success("Face data ready.")
            if st.button("Verify via Face"):
                with st.spinner("Verifying face..."):
                    try:
                        # In a real scenario, the classifier should be loaded/trained beforehand
                        # For demonstration, we assume predict() works if trained
                        predictions = face_pipeline.predict(st.session_state["captured_face"])
                        if predictions:
                            _, label, conf = predictions[0]
                            if label == student_id:
                                db_manager.log_attendance(student_id, subject_id, session_id, "face", conf)
                                st.success(f"Attendance marked! Confidence: {conf:.2f}")
                            else:
                                st.error("Face does not match the provided Student ID.")
                        else:
                            st.error("No face detected.")
                    except Exception as e:
                        st.error(f"Verification failed: {e}")

    with col2:
        if st.button("🎤 Record Voice", use_container_width=True):
            voice_capture_dialog()
            
        if st.session_state["captured_voice"] is not None:
            st.success("Voice data ready.")
            if st.button("Verify via Voice"):
                with st.spinner("Verifying voice..."):
                    try:
                        wav = voice_pipeline.preprocess_audio(st.session_state["captured_voice"])
                        target_emb = voice_pipeline.extract_embedding(wav)
                        
                        # Fetch known embedding from DB (Mocked logic for demonstration)
                        # In production, fetch the specific student's voice_embedding from Supabase
                        st.info("Voice embedding extracted. (Database matching logic executes here)")
                        
                        # Mocking successful log
                        db_manager.log_attendance(student_id, subject_id, session_id, "voice", 0.95)
                        st.success("Attendance marked via Voice!")
                    except Exception as e:
                        st.error(f"Verification failed: {e}")

def render_teacher() -> None:
    """Renders the teacher dashboard for managing sessions and viewing logs."""
    st.title("👨‍🏫 Teacher Dashboard")
    if st.button("⬅️ Back to Home"):
        navigate_to("home")
        
    st.divider()
    
    tab1, tab2 = st.tabs(["Create Session", "View Attendance"])
    
    with tab1:
        st.subheader("Start a New Class Session")
        subject_id = st.text_input("Subject ID (e.g., CS101)")
        session_id = st.text_input("Session ID (e.g., Week-1-Lecture)")
        
        if st.button("Generate Session QR"):
            if subject_id and session_id:
                qr_data = f"snapclass://enroll?subject={subject_id}&session={session_id}"
                render_qr_code(qr_data)
                st.success("Session Active! Students can scan to join.")
            else:
                st.warning("Please enter both Subject ID and Session ID.")
                
    with tab2:
        st.subheader("Attendance Logs")
        view_subject = st.text_input("Subject ID to view", key="view_sub")
        view_session = st.text_input("Session ID to view", key="view_ses")
        
        if st.button("Fetch Logs"):
            with st.spinner("Fetching records..."):
                try:
                    logs = db_manager.get_attendance(view_subject, view_session)
                    if logs:
                        st.dataframe(logs, use_container_width=True)
                    else:
                        st.info("No attendance records found for this session.")
                except Exception as e:
                    st.error(f"Failed to fetch logs: {e}")

# --- Main Router ---
def main() -> None:
    """Main application router."""
    if st.session_state["current_page"] == "home":
        render_home()
    elif st.session_state["current_page"] == "student":
        render_student()
    elif st.session_state["current_page"] == "teacher":
        render_teacher()

if __name__ == "__main__":
    main()
