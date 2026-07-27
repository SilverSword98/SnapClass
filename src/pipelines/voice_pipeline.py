"""
Voice Biometrics Engine for SnapClass.
Utilizes librosa for audio preprocessing (VAD) and Resemblyzer for 
extracting 256D voice embeddings and speaker identification.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import io

import numpy as np
import librosa
from resemblyzer import VoiceEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VoiceRecognitionPipeline:
    """
    Pipeline for processing audio, extracting voice embeddings, and identifying speakers.
    """

    def __init__(self) -> None:
        """
        Initializes the Resemblyzer VoiceEncoder model.
        """
        logger.info("Loading Resemblyzer VoiceEncoder model...")
        try:
            self.encoder = VoiceEncoder()
            self.sampling_rate = 16000  # Resemblyzer expects 16kHz audio
            logger.info("VoiceEncoder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VoiceEncoder: {e}")
            raise

    def preprocess_audio(self, audio_file: io.BytesIO) -> np.ndarray:
        """
        Loads audio from a file-like object, resamples it, and trims silence (VAD).

        Args:
            audio_file (io.BytesIO): The uploaded audio file buffer.

        Returns:
            np.ndarray: The processed audio waveform as a numpy array.
            
        Raises:
            ValueError: If the audio file is invalid or empty.
        """
        try:
            # Load audio and resample to 16kHz
            wav, _ = librosa.load(audio_file, sr=self.sampling_rate)
            
            # Apply Voice Activity Detection (VAD) by trimming silence
            # top_db=30 means silence is anything 30dB below peak volume
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=30)
            
            if len(wav_trimmed) == 0:
                raise ValueError("Audio file contains only silence or is empty.")
                
            return wav_trimmed
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise

    def extract_embedding(self, wav: np.ndarray) -> np.ndarray:
        """
        Extracts a 256D voice embedding from a preprocessed audio waveform.

        Args:
            wav (np.ndarray): The preprocessed audio waveform.

        Returns:
            np.ndarray: A 256-dimensional numpy array representing the voice embedding.
        """
        try:
            embedding = self.encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            logger.error(f"Error extracting voice embedding: {e}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two voice embeddings.

        Args:
            embedding1 (np.ndarray): First voice embedding.
            embedding2 (np.ndarray): Second voice embedding.

        Returns:
            float: Cosine similarity score between -1.0 and 1.0.
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))

    def identify_speaker(
        self, 
        target_embedding: np.ndarray, 
        known_embeddings: Dict[str, np.ndarray], 
        threshold: float = 0.75
    ) -> Tuple[str, float]:
        """
        Identifies the speaker by comparing the target embedding against a dictionary of known embeddings.

        Args:
            target_embedding (np.ndarray): The embedding of the unknown speaker.
            known_embeddings (Dict[str, np.ndarray]): Dictionary mapping student_ids to their embeddings.
            threshold (float): Minimum similarity score required for a positive match.

        Returns:
            Tuple[str, float]: A tuple containing the matched student_id (or "Unknown") and the confidence score.
        """
        best_match = "Unknown"
        highest_score = 0.0

        for student_id, known_emb in known_embeddings.items():
            score = self.compute_similarity(target_embedding, known_emb)
            if score > highest_score:
                highest_score = score
                best_match = student_id

        if highest_score >= threshold:
            return best_match, highest_score
        
        return "Unknown", highest_score
