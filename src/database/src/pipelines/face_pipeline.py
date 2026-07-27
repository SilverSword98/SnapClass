"""
Facial Biometrics Engine for SnapClass.
Utilizes dlib for face detection and 128D feature extraction, 
and scikit-learn Linear SVC for multi-face classification.
"""

import os
import pickle
import logging
from typing import List, Tuple, Optional, Any

import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """
    Pipeline for detecting faces, extracting 128D embeddings, and classifying identities.
    """

    def __init__(self, shape_predictor_path: str, face_rec_model_path: str) -> None:
        """
        Initializes the dlib models and the classifier placeholder.

        Args:
            shape_predictor_path (str): Path to the dlib 68-point shape predictor model.
            face_rec_model_path (str): Path to the dlib face recognition ResNet model.
        
        Raises:
            FileNotFoundError: If the provided dlib model paths do not exist.
        """
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor model not found at {shape_predictor_path}")
        if not os.path.exists(face_rec_model_path):
            raise FileNotFoundError(f"Face recognition model not found at {face_rec_model_path}")

        logger.info("Loading dlib models. This may take a moment...")
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_recognizer = dlib.face_recognition_model_v1(face_rec_model_path)
        
        self.classifier: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None
        logger.info("Dlib models loaded successfully.")

    def extract_embeddings(self, image: np.ndarray) -> List[Tuple[dlib.rectangle, np.ndarray]]:
        """
        Detects faces in an image and extracts their 128D embeddings.

        Args:
            image (np.ndarray): The input image in BGR format (standard OpenCV format).

        Returns:
            List[Tuple[dlib.rectangle, np.ndarray]]: A list of tuples containing the 
            bounding box and the 128D face descriptor as a numpy array.
        """
        # Convert BGR to RGB as dlib expects RGB images
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector(rgb_image, 1)
        results = []

        for rect in detections:
            try:
                # Get facial landmarks
                shape = self.shape_predictor(rgb_image, rect)
                
                # Compute the 128D vector
                face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_image, shape)
                
                # Convert to numpy array
                embedding = np.array(face_descriptor, dtype=np.float32)
                results.append((rect, embedding))
            except Exception as e:
                logger.warning(f"Failed to extract embedding for a detected face: {e}")
                continue

        return results

    def train_classifier(self, embeddings: List[np.ndarray], labels: List[str], save_path: str) -> None:
        """
        Trains a Linear SVC on the provided embeddings and saves the model to disk.

        Args:
            embeddings (List[np.ndarray]): List of 128D face embeddings.
            labels (List[str]): List of corresponding student IDs/labels.
            save_path (str): File path to save the trained model (pickle format).
            
        Raises:
            ValueError: If the number of embeddings and labels do not match, or if data is empty.
        """
        if not embeddings or not labels:
            raise ValueError("Embeddings and labels cannot be empty.")
        if len(embeddings) != len(labels):
            raise ValueError("The number of embeddings must match the number of labels.")

        logger.info(f"Training SVC classifier on {len(embeddings)} samples...")
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Initialize SVC with probability=True to get confidence scores during prediction
        self.classifier = SVC(kernel='linear', probability=True, class_weight='balanced')
        self.classifier.fit(embeddings, encoded_labels)

        # Save the model and label encoder
        model_data = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Classifier trained and saved successfully to {save_path}.")

    def load_classifier(self, model_path: str) -> None:
        """
        Loads a trained SVC classifier and label encoder from disk.

        Args:
            model_path (str): Path to the pickled model file.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If the model file is corrupted or invalid.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data["classifier"]
                self.label_encoder = model_data["label_encoder"]
            logger.info(f"Classifier loaded successfully from {model_path}.")
        except Exception as e:
            logger.error(f"Failed to load classifier from {model_path}: {e}")
            raise

    def predict(self, image: np.ndarray, confidence_threshold: float = 0.6) -> List[Tuple[dlib.rectangle, str, float]]:
        """
        Detects faces in an image and predicts their identities using the trained SVC.

        Args:
            image (np.ndarray): The input image in BGR format.
            confidence_threshold (float): Minimum probability required to assign an identity.

        Returns:
            List[Tuple[dlib.rectangle, str, float]]: A list of tuples containing the 
            bounding box, predicted label (student_id), and confidence score.
            If confidence is below threshold, label will be 'Unknown'.
            
        Raises:
            RuntimeError: If the classifier has not been trained or loaded.
        """
        if self.classifier is None or self.label_encoder is None:
            raise RuntimeError("Classifier is not initialized. Call train_classifier or load_classifier first.")

        extracted_data = self.extract_embeddings(image)
        predictions = []

        for rect, embedding in extracted_data:
            # Reshape for sklearn (1, -1)
            embedding_reshaped = embedding.reshape(1, -1)
            
            # Predict probabilities
            probabilities = self.classifier.predict_proba(embedding_reshaped)[0]
            best_class_index = np.argmax(probabilities)
            best_probability = probabilities[best_class_index]

            if best_probability >= confidence_threshold:
                label = self.label_encoder.inverse_transform([best_class_index])[0]
            else:
                label = "Unknown"

            predictions.append((rect, label, float(best_probability)))

        return predictions
