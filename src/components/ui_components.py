"""
Streamlit UI Components for SnapClass.
Handles QR code generation, camera dialogs, and audio input dialogs.
"""

import io
import cv2
import numpy as np
import segno
import streamlit as st
from typing import Optional


def render_qr_code(data: str, scale: int = 5) -> None:
    """
    Generates and displays a QR code in the Streamlit app.

    Args:
        data (str): The data to encode in the QR code (e.g., a session URL or ID).
        scale (int): The scaling factor for the QR code image size.
    """
    try:
        qr = segno.make_qr(data)
        out = io.BytesIO()
        qr.save(out, kind='png', scale=scale)
        out.seek(0)
        st.image(out, caption="Scan to join class session", use_container_width=False)
    except Exception as e:
        st.error(f"Failed to generate QR code: {e}")


@st.dialog("Capture Facial Biometrics")
def face_capture_dialog() -> None:
    """
    Opens a modal dialog to capture an image using the device camera.
    Saves the captured OpenCV image to Streamlit session state.
    """
    st.write("Please look directly at the camera.")
    camera_file = st.camera_input("Take a picture")
    
    if camera_file is not None:
        # Convert UploadedFile to OpenCV format (BGR)
        bytes_data = camera_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        st.session_state["captured_face"] = cv2_img
        st.success("Face captured successfully!")
        if st.button("Confirm & Close"):
            st.rerun()


@st.dialog("Record Voice Biometrics")
def voice_capture_dialog() -> None:
    """
    Opens a modal dialog to record audio using the device microphone.
    Saves the audio buffer to Streamlit session state.
    """
    st.write("Please read the following phrase clearly:")
    st.info('"My voice is my password, and it verifies my attendance."')
    
    # st.audio_input is available in Streamlit >= 1.36.0
    audio_file = st.audio_input("Record your voice")
    
    if audio_file is not None:
        st.session_state["captured_voice"] = audio_file
        st.success("Voice recorded successfully!")
        if st.button("Confirm & Close"):
            st.rerun()
