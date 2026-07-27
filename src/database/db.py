"""
Database Abstraction Layer for SnapClass.
Handles interactions with Supabase for student management, enrollments, and attendance.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from postgrest.exceptions import APIError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database operations using the Supabase Python SDK.
    """

    def __init__(self) -> None:
        """
        Initializes the Supabase client using environment variables.
        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_KEY are not set in the environment.
        """
        supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
        supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials missing from environment variables.")
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")

        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("Successfully initialized Supabase client.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    def create_student(
        self, 
        student_id: str, 
        full_name: str, 
        face_embedding: Optional[List[float]] = None, 
        voice_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Registers a new student in the database with optional biometric embeddings.

        Args:
            student_id (str): Unique identifier for the student (e.g., matriculation number).
            full_name (str): Full name of the student.
            face_embedding (Optional[List[float]]): 128D facial descriptor.
            voice_embedding (Optional[List[float]]): Voice embedding vector.

        Returns:
            Dict[str, Any]: The inserted student record.
        """
        data = {
            "student_id": student_id,
            "full_name": full_name,
            "face_embedding": face_embedding,
            "voice_embedding": voice_embedding
        }
        
        try:
            response = self.client.table("students").insert(data).execute()
            logger.info(f"Successfully created student: {student_id}")
            return response.data[0]
        except APIError as e:
            logger.error(f"Supabase API Error creating student {student_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating student {student_id}: {e}")
            raise

    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Retrieves all registered students. Useful for training biometric classifiers.

        Returns:
            List[Dict[str, Any]]: A list of all student records.
        """
        try:
            response = self.client.table("students").select("*").execute()
            return response.data
        except APIError as e:
            logger.error(f"Supabase API Error fetching all students: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching all students: {e}")
            raise

    def enroll_student(self, student_id: str, subject_id: str) -> Dict[str, Any]:
        """
        Enrolls a student in a specific subject.

        Args:
            student_id (str): The unique student identifier.
            subject_id (str): The unique subject identifier.

        Returns:
            Dict[str, Any]: The enrollment record.
        """
        data = {
            "student_id": student_id,
            "subject_id": subject_id
        }
        try:
            response = self.client.table("enrollments").insert(data).execute()
            logger.info(f"Enrolled student {student_id} in subject {subject_id}")
            return response.data[0]
        except APIError as e:
            logger.error(f"Supabase API Error enrolling student {student_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error enrolling student {student_id}: {e}")
            raise

    def log_attendance(
        self, 
        student_id: str, 
        subject_id: str, 
        session_id: str, 
        method: str, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Logs an attendance record for a student.

        Args:
            student_id (str): The unique student identifier.
            subject_id (str): The unique subject identifier.
            session_id (str): The specific class session identifier.
            method (str): The biometric method used ('face' or 'voice').
            confidence (float): The confidence score of the biometric prediction.

        Returns:
            Dict[str, Any]: The created attendance log.
        """
        if method not in ["face", "voice"]:
            raise ValueError("Attendance method must be either 'face' or 'voice'.")

        data = {
            "student_id": student_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "verification_method": method,
            "confidence_score": confidence
        }
        try:
            response = self.client.table("attendance_logs").insert(data).execute()
            logger.info(f"Logged {method} attendance for {student_id} in session {session_id}")
            return response.data[0]
        except APIError as e:
            logger.error(f"Supabase API Error logging attendance for {student_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error logging attendance for {student_id}: {e}")
            raise

    def get_attendance(self, subject_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all attendance logs for a specific subject session.

        Args:
            subject_id (str): The unique subject identifier.
            session_id (str): The specific class session identifier.

        Returns:
            List[Dict[str, Any]]: A list of attendance records.
        """
        try:
            response = (
                self.client.table("attendance_logs")
                .select("*, students(full_name)")
                .eq("subject_id", subject_id)
                .eq("session_id", session_id)
                .execute()
            )
            return response.data
        except APIError as e:
            logger.error(f"Supabase API Error fetching attendance for session {session_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching attendance for session {session_id}: {e}")
            raise
