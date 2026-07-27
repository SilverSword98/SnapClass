# 📸 SnapClass: AI-Powered Biometric Attendance System

SnapClass is a multi-modal, AI-powered classroom attendance application. It leverages facial recognition and voice biometrics to securely and efficiently log student attendance, replacing traditional roll calls with a seamless Streamlit interface.

## 🚀 Features
* **Multi-Role Dashboards:** Distinct routing and UI flows for Students and Teachers.
* **Facial Biometrics:** Uses `dlib` (68-point shape predictor & 128D ResNet descriptor) and a Scikit-Learn Linear SVC for multi-face recognition.
* **Voice Biometrics:** Uses `librosa` for Voice Activity Detection (VAD) and `resemblyzer` for 256D voice embedding extraction and cosine-similarity matching.
* **Cloud Database:** Integrated with Supabase (PostgreSQL) for student CRUD, subject enrollments, and attendance logging.
* **Dynamic QR Codes:** Teachers can generate session-specific QR codes for students to scan and join.

## 🛠️ Technology Stack
* **Frontend:** Streamlit, Segno (QR generation)
* **Computer Vision:** OpenCV, Dlib, Scikit-Learn
* **Audio Processing:** Librosa, Resemblyzer
* **Database & Auth:** Supabase Python SDK

## 📂 Project Architecture


## 🧠 System Architecture

```mermaid
graph TD
subgraph Frontend
UI[Streamlit Web App]
QR[Segno QR Generator]
end

subgraph Biometric AI Engine
FACE[Face Pipeline: Dlib + SVC]
VOICE[Voice Pipeline: Resemblyzer + Librosa]
end

subgraph Cloud Database
DB[(Supabase PostgreSQL)]
VEC[Vector Embeddings]
LOGS[Attendance Logs]
end

UI -->|Captures Image/Audio| FACE
UI -->|Captures Image/Audio| VOICE
UI -->|Generates Session| QR

FACE -->|Extracts 128D Features| VEC
VOICE -->|Extracts 256D Features| VEC

FACE -->|Verifies Identity| DB
VOICE -->|Verifies Identity| DB
DB -->|Records Success| LOGS
```

## 🔄 Biometric Verification Flow

```mermaid
sequenceDiagram
participant Student
participant UI as Streamlit App
participant AI as ML Pipeline
participant DB as Supabase

Student->>UI: Enters ID & Captures Face/Voice
UI->>AI: Sends Media Buffer
AI->>AI: Extracts Embeddings (128D/256D)
AI->>DB: Fetches Known Student Embeddings
AI-->>UI: Returns Confidence Score

alt Score >= Threshold (e.g., 0.75)
UI->>DB: Logs Attendance Record
UI-->>Student: ✅ Success: Attendance Marked
else Score < Threshold
UI-->>Student: ❌ Failed: Identity Not Verified
end
```

```text
snapclass/
├── app.py                              # Main Streamlit application & router
├── requirements.txt                    # Python dependencies
├── .env                                # Supabase credentials (ignored in git)
├── models/                             # Dlib model weights (ignored in git)
└── src/
    ├── components/
    │   └── ui_components.py            # Streamlit modal dialogs & QR rendering
    ├── database/
    │   └── db.py                       # Supabase abstraction layer
    └── pipelines/
        ├── face_pipeline.py            # Dlib + SVC facial recognition engine
        └── voice_pipeline.py           # Resemblyzer audio processing engine




