"""
Face Recognition Engine Module

This module provides core functionality for face detection, encoding,
and comparison using the DeepFace library.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Install with: pip install deepface")


def detect_faces(frame: np.ndarray, detector_backend: str = "opencv") -> List[Dict[str, Any]]:
    """
    Detect faces in a frame.
    
    Args:
        frame: BGR image from OpenCV
        detector_backend: "opencv", "ssd", "mtcnn", "retinaface", etc.
    
    Returns:
        List of face detection results with facial_area and confidence
    """
    if not DEEPFACE_AVAILABLE:
        return []
    
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        
        # Filter out low confidence detections
        valid_faces = [f for f in faces if f.get('confidence', 0) > 0.5]
        return valid_faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def get_face_locations(frame: np.ndarray, detector_backend: str = "opencv") -> List[Tuple[int, int, int, int]]:
    """
    Get face bounding box locations.
    
    Args:
        frame: BGR image from OpenCV
        detector_backend: Detection backend to use
    
    Returns:
        List of (top, right, bottom, left) tuples
    """
    faces = detect_faces(frame, detector_backend)
    locations = []
    
    for face in faces:
        area = face.get('facial_area', {})
        x = area.get('x', 0)
        y = area.get('y', 0)
        w = area.get('w', 0)
        h = area.get('h', 0)
        
        # Convert to (top, right, bottom, left) format
        top = y
        right = x + w
        bottom = y + h
        left = x
        
        locations.append((top, right, bottom, left))
    
    return locations


def get_face_encoding(frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None, 
                      model_name: str = "Facenet512") -> Optional[np.ndarray]:
    """
    Get face embedding from a frame.
    
    Args:
        frame: BGR image from OpenCV
        face_location: Optional (top, right, bottom, left) tuple to crop face
        model_name: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace", etc.
    
    Returns:
        Face embedding array or None if no face found
    """
    if not DEEPFACE_AVAILABLE:
        return None
    
    try:
        # If face_location is provided, crop the face region
        if face_location is not None:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]
            
            # Add padding
            pad = 20
            y1 = max(0, top - pad)
            y2 = min(h, bottom + pad)
            x1 = max(0, left - pad)
            x2 = min(w, right + pad)
            
            frame = frame[y1:y2, x1:x2]
            
            if frame.size == 0:
                return None
        
        # DeepFace.represent returns embeddings
        result = DeepFace.represent(
            img_path=frame,
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=False
        )
        
        if result and len(result) > 0:
            embedding = result[0].get('embedding', None)
            if embedding:
                return np.array(embedding)
        return None
    except Exception as e:
        print(f"Encoding error: {e}")
        return None


def compare_faces(known_encodings: List[np.ndarray], face_encoding: np.ndarray, 
                  threshold: float = 0.4) -> List[bool]:
    """
    Compare a face encoding against known encodings using cosine distance.
    
    Args:
        known_encodings: List of known face encodings
        face_encoding: Face encoding to compare
        threshold: Distance threshold (lower = stricter, default for Facenet512)
    
    Returns:
        List of boolean matches
    """
    if len(known_encodings) == 0 or face_encoding is None:
        return []
    
    matches = []
    for known_enc in known_encodings:
        distance = cosine_distance(known_enc, face_encoding)
        matches.append(distance <= threshold)
    
    return matches


def cosine_distance(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """Calculate cosine distance between two encodings."""
    dot = np.dot(encoding1, encoding2)
    norm1 = np.linalg.norm(encoding1)
    norm2 = np.linalg.norm(encoding2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    similarity = dot / (norm1 * norm2)
    distance = 1 - similarity
    return distance


def calculate_face_distance(known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> np.ndarray:
    """
    Calculate distance between a face encoding and known encodings.
    
    Args:
        known_encodings: List of known face encodings
        face_encoding: Face encoding to compare
    
    Returns:
        Array of distances (lower = more similar)
    """
    if len(known_encodings) == 0 or face_encoding is None:
        return np.array([])
    
    distances = []
    for known_enc in known_encodings:
        dist = cosine_distance(known_enc, face_encoding)
        distances.append(dist)
    
    return np.array(distances)


def find_best_match(known_encodings: List[np.ndarray], known_names: List[str], 
                    face_encoding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
    """
    Find the best matching face from known encodings.
    
    Args:
        known_encodings: List of known face encodings
        known_names: List of corresponding names
        face_encoding: Face encoding to match
        threshold: Distance threshold (lower = stricter)
    
    Returns:
        Tuple of (name, confidence) where confidence is 1 - distance
    """
    if len(known_encodings) == 0 or face_encoding is None:
        return ("Unknown", 0.0)
    
    distances = calculate_face_distance(known_encodings, face_encoding)
    
    if len(distances) == 0:
        return ("Unknown", 0.0)
    
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]
    
    if best_distance <= threshold:
        confidence = 1 - best_distance
        return (known_names[best_match_index], min(confidence, 1.0))
    else:
        return ("Unknown", 0.0)


def draw_face_box(frame: np.ndarray, face_location: Tuple[int, int, int, int], 
                  name: str, confidence: float = 0.0, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw a bounding box and label on a face.
    
    Args:
        frame: BGR image from OpenCV
        face_location: (top, right, bottom, left) tuple
        name: Name to display
        confidence: Confidence score (0-1)
        color: BGR color tuple
    
    Returns:
        Frame with drawn box and label
    """
    frame = frame.copy()
    top, right, bottom, left = face_location
    
    # Draw rectangle
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Prepare label
    if confidence > 0:
        label = f"{name} ({confidence:.1%})"
    else:
        label = name
    
    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                  (left + label_size[0] + 10, top), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (left + 5, top - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def draw_guide_overlay(frame: np.ndarray, direction: str, progress: int = 0) -> np.ndarray:
    """
    Draw guidance overlay for face registration.
    
    Args:
        frame: BGR image from OpenCV
        direction: "center", "left", "right", "up", "down"
        progress: Current progress (0-5)
    
    Returns:
        Frame with overlay
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw semi-transparent overlay at top
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Direction instructions
    instructions = {
        "center": "Hadapkan wajah ke depan",
        "left": "Tolehkan wajah ke KIRI",
        "right": "Tolehkan wajah ke KANAN",
        "up": "Tengadahkan wajah ke ATAS",
        "down": "Tundukkan wajah ke BAWAH"
    }
    
    text = instructions.get(direction, "")
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw progress indicator
    progress_text = f"Progress: {progress}/5"
    cv2.putText(frame, progress_text, (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw direction arrow
    center_x, center_y = w // 2, h // 2
    arrow_length = 80
    
    if direction == "left":
        cv2.arrowedLine(frame, (center_x + 50, center_y), (center_x - arrow_length, center_y), (0, 255, 255), 3, tipLength=0.3)
    elif direction == "right":
        cv2.arrowedLine(frame, (center_x - 50, center_y), (center_x + arrow_length, center_y), (0, 255, 255), 3, tipLength=0.3)
    elif direction == "up":
        cv2.arrowedLine(frame, (center_x, center_y + 50), (center_x, center_y - arrow_length), (0, 255, 255), 3, tipLength=0.3)
    elif direction == "down":
        cv2.arrowedLine(frame, (center_x, center_y - 50), (center_x, center_y + arrow_length), (0, 255, 255), 3, tipLength=0.3)
    elif direction == "center":
        # Draw crosshair
        cv2.circle(frame, (center_x, center_y), 60, (0, 255, 255), 2)
        cv2.line(frame, (center_x - 80, center_y), (center_x + 80, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 80), (center_x, center_y + 80), (0, 255, 255), 1)
    
    return frame


def get_all_face_encodings(frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                           model_name: str = "Facenet512") -> List[np.ndarray]:
    """
    Get face encodings for multiple detected faces by cropping and encoding each.
    
    Args:
        frame: BGR image from OpenCV
        face_locations: List of (top, right, bottom, left) tuples
        model_name: Model to use for encoding
    
    Returns:
        List of face encoding arrays
    """
    if not DEEPFACE_AVAILABLE:
        return []
    
    encodings = []
    
    for top, right, bottom, left in face_locations:
        # Crop face region with some padding
        h, w = frame.shape[:2]
        pad = 20
        y1 = max(0, top - pad)
        y2 = min(h, bottom + pad)
        x1 = max(0, left - pad)
        x2 = min(w, right + pad)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size > 0:
            encoding = get_face_encoding(face_img, model_name=model_name)
            if encoding is not None:
                encodings.append(encoding)
    
    return encodings
