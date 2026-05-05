import os
import json
import argparse
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANVAS_W, CANVAS_H, SEED, DATA_DIR, ONE_FACE_DIR, TWO_FACES_DIR
from dataset.utils import get_hsv_from_z, make_mouth_path, convert_svgs_to_pngs

def get_face_params(z, face_type="one_face"):
    if face_type == "one_face":
        r_b = (55, 75)
        er_b = (4, 10)
        es_b = (15, 30)
        ey_b = (10, 25)
        mw_b = (25, 50)
        my_b = (15, 30)
        mc_b = (-20, 20)
    else: # two_faces
        r_b = (25, 45)
        er_b = (3, 7)
        es_b = (10, 20)
        ey_b = (5, 15)
        mw_b = (15, 30)
        my_b = (10, 20)
        mc_b = (-15, 15)

    face_radius = r_b[0] + z[0] * (r_b[1] - r_b[0])
    
    start_cx, end_cx = face_radius + 10, CANVAS_W - face_radius - 10
    cx = start_cx + z[1] * (end_cx - start_cx)
    
    start_cy, end_cy = face_radius + 10, CANVAS_H - face_radius - 10
    cy = start_cy + z[2] * (end_cy - start_cy)
    
    eye_radius = er_b[0] + z[3] * (er_b[1] - er_b[0])
    eye_spacing = es_b[0] + z[4] * (es_b[1] - es_b[0])
    eye_y_offset = ey_b[0] + z[5] * (ey_b[1] - ey_b[0])
    
    mouth_width = mw_b[0] + z[6] * (mw_b[1] - mw_b[0])
    mouth_y_offset = my_b[0] + z[7] * (my_b[1] - my_b[0])
    mouth_curve = mc_b[0] + z[8] * (mc_b[1] - mc_b[0])
    
    return face_radius, cx, cy, eye_radius, eye_spacing, eye_y_offset, mouth_width, mouth_y_offset, mouth_curve

def generate_face_svg_group(z, face_id, face_type="one_face"):
    face_radius, cx, cy, eye_radius, eye_spacing, eye_y_offset, mouth_width, mouth_y_offset, mouth_curve = get_face_params(z, face_type)
    
    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset
    
    skin = get_hsv_from_z(z[9:12])
    eye = get_hsv_from_z(z[12:15])
    stroke_color = "#111111"
    
    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)
    
    svg_group = f'''  <g id="{face_id}">
    <circle id="{face_id}_base" cx="{cx}" cy="{cy}" r="{face_radius}" fill="{skin['hex']}" stroke="{stroke_color}" stroke-width="2" />
    <circle id="{face_id}_eye_left" cx="{left_eye_cx}" cy="{left_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <circle id="{face_id}_eye_right" cx="{right_eye_cx}" cy="{right_eye_cy}" r="{eye_radius}" fill="{eye['hex']}" />
    <path id="{face_id}_mouth" d="{mouth_path}" fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" />
  </g>'''

    # Keep backward compatibility with metadata parts array format exactly
    if face_type == "one_face":
        # original one_face script saved them without prefix except id
        metadata = {
            "id": face_id,
            "canvas": {"width": CANVAS_W, "height": CANVAS_H},
            "colors": {"skin": skin, "eyes": eye},
            "parts": [
                {"id": "face_base", "type": "circle", "center": [cx, cy], "radius": face_radius},
                {"id": "eye_left", "type": "circle", "center": [left_eye_cx, left_eye_cy], "radius": eye_radius},
                {"id": "eye_right", "type": "circle", "center": [right_eye_cx, right_eye_cy], "radius": eye_radius},
                {"id": "mouth", "type": "path", "center": [mouth_cx, mouth_cy], "width": mouth_width, "curve": mouth_curve}
            ]
        }
    else:
        # original two_face saved them with prefix
        metadata = {
            "face_id": face_id,
            "colors": {"skin": skin, "eyes": eye},
            "parts": [
                {"id": f"{face_id}_base", "type": "circle", "center": [cx, cy], "radius": face_radius},
                {"id": f"{face_id}_eye_left", "type": "circle", "center": [left_eye_cx, left_eye_cy], "radius": eye_radius},
                {"id": f"{face_id}_eye_right", "type": "circle", "center": [right_eye_cx, right_eye_cy], "radius": eye_radius},
                {"id": f"{face_id}_mouth", "type": "path", "center": [mouth_cx, mouth_cy], "width": mouth_width, "curve": mouth_curve}
            ]
        }
    return svg_group, metadata, face_radius, cx, cy

def build_dataset(num_faces=1, n_total=10000, out_dir=DATA_DIR):
    np.random.seed(SEED)
    
    # Ensure the main output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    if num_faces == 1:
        base_dir = os.path.join(out_dir, "svg_face_dataset_one_face")
        z_filename = "Z_10k_one_face.npy"
        z_dim = 15
    else:
        base_dir = os.path.join(out_dir, "svg_face_dataset_two_faces")
        z_filename = "Z_10k_two_faces.npy"
        z_dim = 30
        
    svg_dir = os.path.join(base_dir, "svgs")
    meta_dir = os.path.join(base_dir, "meta")
    png_dir = os.path.join(base_dir, "pngs")
    
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Random seeds for backwards reproducibility
    if num_faces == 1:
        seeds = np.random.choice(10 * n_total, n_total, replace=False)
    
    Z = np.zeros((n_total, z_dim))
    
    print(f"Generating {n_total} SVGs ({num_faces} face(s) per image) in {base_dir}...")
    for i in range(n_total):
        if num_faces == 1:
            sample_id = f"single_face_{i:05d}"
            # Emulate the exact seed usage from the original file for one_face
            np.random.seed(seeds[i])
            z_full = np.random.uniform(0, 1, 15)
            
            svg_group, metadata, _, _, _ = generate_face_svg_group(z_full, sample_id, face_type="one_face")
            
            # The one_face metadata was exactly the returned dict, just overwrite id properly
            metadata["id"] = sample_id
            
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
{svg_group}
</svg>'''
        else:
            sample_id = f"two_faces_{i:05d}"
            while True:
                z_full = np.random.uniform(0, 1, 30)
                
                # Check overlap
                r1, cx1, cy1 = get_face_params(z_full[0:15], "two_faces")[:3]
                r2, cx2, cy2 = get_face_params(z_full[15:30], "two_faces")[:3]
                
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                min_dist = r1 + r2 + 5
                if dist_sq >= min_dist**2:
                    break
                    
            svg1, meta1, _, _, _ = generate_face_svg_group(z_full[0:15], "face_1", "two_faces")
            svg2, meta2, _, _, _ = generate_face_svg_group(z_full[15:30], "face_2", "two_faces")
            
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
{svg1}
{svg2}
</svg>'''
            metadata = {
                "id": sample_id,
                "canvas": {"width": CANVAS_W, "height": CANVAS_H},
                "faces": [meta1, meta2]
            }

        Z[i] = z_full
        
        with open(os.path.join(svg_dir, f"{sample_id}.svg"), "w", encoding="utf-8") as f:
            f.write(svg_content)
        with open(os.path.join(meta_dir, f"{sample_id}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n_total} faces...")
            
    np.save(os.path.join(out_dir, z_filename), Z)
    print(f"Success! Dataset generated and {z_filename} saved.")
    
    # We will skip automatic conversion here to let the user control it if they want
    # convert_svgs_to_pngs(svg_dir, png_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVG Face Dataset")
    parser.add_argument("--faces", type=int, choices=[1, 2], default=1, help="Number of faces per image")
    parser.add_argument("--samples", type=int, default=10000, help="Total number of samples to generate")
    parser.add_argument("--out_dir", type=str, default=DATA_DIR, help="Output directory base")
    parser.add_argument("--convert", action="store_true", help="Convert SVGs to PNGs immediately after generation")
    args = parser.parse_args()
    
    build_dataset(num_faces=args.faces, n_total=args.samples, out_dir=args.out_dir)
    
    if args.convert:
        base_dir = os.path.join(args.out_dir, f"svg_face_dataset_{'one_face' if args.faces == 1 else 'two_faces'}")
        convert_svgs_to_pngs(os.path.join(base_dir, "svgs"), os.path.join(base_dir, "pngs"))
