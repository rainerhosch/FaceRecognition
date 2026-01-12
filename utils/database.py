"""
Database Module for Face Data Storage

This module handles saving, loading, and managing face encodings
using pickle files for local storage.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

# Default data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faces")


def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def get_face_file_path(name: str) -> str:
    """Get the file path for a person's face data."""
    # Sanitize name for filename
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    return os.path.join(DATA_DIR, f"{safe_name}.pkl")


def save_face_data(name: str, encodings: List[np.ndarray], overwrite: bool = False) -> bool:
    """
    Save face encodings for a person.
    
    Args:
        name: Person's name
        encodings: List of face encodings (128-d arrays)
        overwrite: Whether to overwrite existing data
    
    Returns:
        True if successful, False otherwise
    """
    ensure_data_dir()
    file_path = get_face_file_path(name)
    
    if os.path.exists(file_path) and not overwrite:
        # Append to existing encodings
        existing_data = load_single_face_data(name)
        if existing_data:
            encodings = existing_data["encodings"] + encodings
    
    data = {
        "name": name,
        "encodings": encodings,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "num_samples": len(encodings)
    }
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving face data: {e}")
        return False


def load_single_face_data(name: str) -> Optional[Dict]:
    """
    Load face data for a specific person.
    
    Args:
        name: Person's name
    
    Returns:
        Dictionary with face data or None if not found
    """
    file_path = get_face_file_path(name)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading face data for {name}: {e}")
        return None


def load_all_face_data() -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all face data from storage.
    
    Returns:
        Tuple of (all_encodings, all_names) where each encoding maps to a name
    """
    ensure_data_dir()
    
    all_encodings = []
    all_names = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pkl'):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    name = data["name"]
                    for encoding in data["encodings"]:
                        all_encodings.append(encoding)
                        all_names.append(name)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return all_encodings, all_names


def delete_face_data(name: str) -> bool:
    """
    Delete face data for a person.
    
    Args:
        name: Person's name
    
    Returns:
        True if successful, False otherwise
    """
    file_path = get_face_file_path(name)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting face data: {e}")
            return False
    return False


def list_registered_faces() -> List[Dict]:
    """
    List all registered faces.
    
    Returns:
        List of dictionaries with face metadata
    """
    ensure_data_dir()
    faces = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pkl'):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    faces.append({
                        "name": data["name"],
                        "num_samples": data.get("num_samples", len(data["encodings"])),
                        "created_at": data.get("created_at", "Unknown"),
                        "updated_at": data.get("updated_at", "Unknown")
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return faces


def get_registered_count() -> int:
    """Get the count of registered faces."""
    ensure_data_dir()
    return len([f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')])


# Attendance logging functions
ATTENDANCE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "attendance.csv")


def log_attendance(name: str, confidence: float) -> bool:
    """
    Log an attendance entry.
    
    Args:
        name: Person's name
        confidence: Recognition confidence
    
    Returns:
        True if successful
    """
    ensure_data_dir()
    
    try:
        # Create file with header if it doesn't exist
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w') as f:
                f.write("timestamp,name,confidence\n")
        
        # Append entry
        with open(ATTENDANCE_FILE, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{name},{confidence:.4f}\n")
        
        return True
    except Exception as e:
        print(f"Error logging attendance: {e}")
        return False


def get_attendance_log() -> List[Dict]:
    """
    Get all attendance entries.
    
    Returns:
        List of attendance records
    """
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    
    entries = []
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    entries.append({
                        "timestamp": parts[0],
                        "name": parts[1],
                        "confidence": float(parts[2])
                    })
    except Exception as e:
        print(f"Error reading attendance log: {e}")
    
    return entries


def clear_attendance_log() -> bool:
    """Clear the attendance log."""
    try:
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w') as f:
                f.write("timestamp,name,confidence\n")
        return True
    except Exception as e:
        print(f"Error clearing attendance log: {e}")
        return False
