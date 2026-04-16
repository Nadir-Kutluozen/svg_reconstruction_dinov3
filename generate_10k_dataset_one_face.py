import os
import random
import colorsys
import json
from generate_svg import convert_svgs_to_pngs

# --- PATH SETUP ---
BASE_DIR = "svg_face_dataset_one_face"
SVG_DIR = os.path.join(BASE_DIR, "svgs")
META_DIR = os.path.join(BASE_DIR, "meta")
PNG_DIR = os.path.join(BASE_DIR, "pngs")

os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

CANVAS_W = 224
CANVAS_H = 224

def get_random_hsv():
    """Returns a tuple of (h, s, v) and the hex string for internal use."""
    h = random.random()
    s = random.uniform(0.5, 1.0) # Vibrant range
    v = random.uniform(0.5, 1.0) 
    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    hex_val = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    
    return {
        "hsv": [h, s, v],
        "rgb": [r, g, b],
        "hex": hex_val
    }

def make_mouth_path(cx, cy, width, curve):
    """Creates a Quadratic Bezier path for the mouth based on curve value."""
    x1 = cx - width / 2
    x2 = cx + width / 2
    control_y = cy + curve
    # M = Move to start, Q = Quadratic Bezier (control point, end point)
    return f"M {x1:.2f} {cy:.2f} Q {cx:.2f} {control_y:.2f} {x2:.2f} {cy:.2f}"

def generate_face_svg(sample_id):
    # --- Geometry ---
    face_radius = random.randint(55, 75)
    cx = random.randint(face_radius + 10, CANVAS_W - face_radius - 10)
    cy = random.randint(face_radius + 10, CANVAS_H - face_radius - 10)

    eye_radius = random.randint(4, 10)
    eye_spacing = random.randint(15, 30)
    eye_y_offset = random.randint(10, 25)

    mouth_width = random.randint(25, 50)
    mouth_y_offset = random.randint(15, 30)
    mouth_curve = random.randint(-20, 20) 

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    # --- Independent Colors for Interpretability ---
    skin = get_random_hsv()
    eye = get_random_hsv()
    stroke_color = "#111111"

    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)

    # --- SVG Generation ---
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
  <g id="face">
    <circle id="face_base" cx="{cx}" cy="{cy}" r="{face_radius}" fill="{skin['hex']}" stroke="{stroke_color}" stroke-width="2" />
    <circle id="eye_left" cx="{left_eye_cx}" cy="{left_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <circle id="eye_right" cx="{right_eye_cx}" cy="{right_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <path id="mouth" d="{mouth_path}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" />
  </g>
</svg>'''

    # --- Metadata (Probing Target) ---
    metadata = {
        "id": sample_id,
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "colors": {
            "skin": {
                "hsv": skin["hsv"],
                "rgb": skin["rgb"]
            },
            "eyes": {
                "hsv": eye["hsv"],
                "rgb": eye["rgb"]
            }
        },
        "parts": [
            {"id": "face_base", "type": "circle", "center": [cx, cy], "radius": face_radius},
            {"id": "eye_left", "type": "circle", "center": [left_eye_cx, left_eye_cy], "radius": eye_radius},
            {"id": "eye_right", "type": "circle", "center": [right_eye_cx, right_eye_cy], "radius": eye_radius},
            {"id": "mouth", "type": "path", "center": [mouth_cx, mouth_cy], "width": mouth_width, "curve": mouth_curve}
        ]
    }

    return svg, metadata

if __name__ == "__main__":
    # --- EXECUTION ---
    n_total = 10000

    for i in range(n_total):
        sample_id = f"single_face_{i:05d}"
        svg_content, meta_data = generate_face_svg(sample_id)

        # File paths
        svg_path = os.path.join(SVG_DIR, f"{sample_id}.svg")
        meta_path = os.path.join(META_DIR, f"{sample_id}.json")

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    print(f"Successfully generated/overwrote {n_total} files in {BASE_DIR}")

    # Run PNG conversion over all generated SVGs
    convert_svgs_to_pngs(SVG_DIR, PNG_DIR)
