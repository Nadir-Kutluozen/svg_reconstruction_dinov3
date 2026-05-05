# Constants and configurations for the project
import os

# Base directory for the project (assumes we are running from project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the central data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define the central output directory for plots
OUTPUT_DIR = os.path.join(BASE_DIR, "output_pngs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Specific data paths
ONE_FACE_DIR = os.path.join(DATA_DIR, "svg_face_dataset_one_face")
TWO_FACES_DIR = os.path.join(DATA_DIR, "svg_face_dataset_two_faces")

SEED = 42
CANVAS_W = 224
CANVAS_H = 224

ONE_FACE_LABELS = [
    "face_radius", "face_cx", "face_cy", 
    "eye_radius", "eye_spacing", "eye_y_offset", 
    "mouth_width", "mouth_y_offset", "mouth_curve",
    "skin_h", "skin_s", "skin_v",
    "eye_h", "eye_s", "eye_v"
]

TWO_FACES_LABELS = [f"f1_{L}" for L in ONE_FACE_LABELS] + [f"f2_{L}" for L in ONE_FACE_LABELS]
