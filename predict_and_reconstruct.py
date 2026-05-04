import os
import io
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cairosvg
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linear_probe_functions import load_dino_features, load_metadata_features, extract_dino_features
from generate_svg import make_mouth_path, CANVAS_W, CANVAS_H

SEED = 42

def build_svg_from_predictions(preds):
    """
    Takes a 15-dimensional list/array of predicted z-values and builds an SVG string.
    Features (0 to 1 range): face_radius, cx, cy, eye_radius, eye_spacing, eye_y_offset, 
                             mouth_width, mouth_y_offset, mouth_curve,
                             skin_h, skin_s, skin_v, eye_h, eye_s, eye_v
    """
    import colorsys
    # Clip predictions to roughly [0, 1] range to avoid extreme deformations
    z = np.clip(preds, 0.0, 1.0)
    
    # Decode geometry
    start, end = 55, 75
    face_radius = start + z[0] * (end - start)
    
    start, end = face_radius + 10, CANVAS_W - face_radius - 10
    cx = start + z[1] * (end - start)
    
    start, end = face_radius + 10, CANVAS_H - face_radius - 10
    cy = start + z[2] * (end - start)

    start, end = 4, 10
    eye_radius = start + z[3] * (end - start)
    
    start, end = 15, 30
    eye_spacing = start + z[4] * (end - start)
    
    start, end = 10, 25
    eye_y_offset = start + z[5] * (end - start)

    start, end = 25, 50
    mouth_width = start + z[6] * (end - start)
    
    start, end = 15, 30
    mouth_y_offset = start + z[7] * (end - start)

    start, end = -20, 20
    mouth_curve = start + z[8] * (end - start)

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    # Decode colors
    def get_hex(h, s_z, v_z):
        s = 0.5 + s_z * 0.5
        v = 0.5 + v_z * 0.5
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

    skin_hex = get_hex(z[9], z[10], z[11])
    eye_hex = get_hex(z[12], z[13], z[14])
    
    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <rect width="100%" height="100%" fill="white" />
  <g>
    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{face_radius:.2f}" fill="{skin_hex}" stroke="#111111" stroke-width="2" />
    <circle cx="{left_eye_cx:.2f}" cy="{left_eye_cy:.2f}" r="{eye_radius:.2f}" fill="{eye_hex}" />
    <circle cx="{right_eye_cx:.2f}" cy="{right_eye_cy:.2f}" r="{eye_radius:.2f}" fill="{eye_hex}" />
    <path d="{mouth_path}" fill="none" stroke="#111111" stroke-width="3" stroke-linecap="round" />
  </g>
</svg>'''
    return svg

def render_svg_to_pil(svg_str):
    """Converts a raw SVG string into a PIL Image for plotting natively."""
    png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=224, output_height=224)
    return Image.open(io.BytesIO(png_data))

def plot_reconstruction_grid(image_ids, source_dir, predicted_features, output_filename, title_prefix=""):
    """
    Plots a side-by-side grid of original vs. reconstructed images.
    """
    num_images = len(image_ids)
    if num_images == 0:
        print("No images provided for plotting.")
        return
        
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(num_images):
        img_id = image_ids[i]
        orig_img_path = os.path.join(source_dir, f"{img_id}.png")
        if not os.path.exists(orig_img_path):
             # just in case user file drops extension or has variation
             orig_img_path = os.path.join(source_dir, f"{img_id}")
        
        orig_pil = Image.open(orig_img_path).convert("RGB")
        
        predicted_svg_str = build_svg_from_predictions(predicted_features[i])
        predicted_pil = render_svg_to_pil(predicted_svg_str)
        
        axes[i, 0].imshow(orig_pil)
        # Hide ticks but keep the border box
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        axes[i, 1].imshow(predicted_pil)
        # Hide ticks but keep the border box
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        for spine in axes[i, 1].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"Saved side-by-side reconstruction to {output_filename}")


def reconstruct_unseen_in_distribution(probe, X_test, ids_test):
    """Randomly selects 5 images from the Test set and reconstructs them."""
    print("Reconstructing 5 random IN-DISTRIBUTION (white bg) test images...")
    
    # Randomly select 5 indices from the test set
    np.random.seed() # local randomize
    indices = np.random.choice(len(X_test), 5, replace=False) # 5 images from the test set
    
    sample_ids = [ids_test[i] for i in indices]
    sample_X = X_test[indices]
    
    # Predict
    preds = probe.predict(sample_X)
    
    # Plot
    plot_reconstruction_grid(
        sample_ids, 
        "svg_face_dataset_one_face/pngs", 
        preds, 
        "reconstruction_in_distribution.png", 
        "In-Distribution"
    )

def reconstruct_emojis(probe):
    """Extracts features for out-of-distribution OOD emojis and reconstructs them."""
    input_dir = "input_pngs"
    print(f"Extracting DINOv3 representations from {input_dir}...")
    
    input_ids, input_X_layers = extract_dino_features(input_dir, pretrained=True)
    if len(input_ids) == 0:
        return
        
    X_input = input_X_layers[:, -1, :]
    print("Predicting SVG structural values for OOD inputs...")
    preds = probe.predict(X_input)
    
    # Append .png to ids manually here for compatibility since input_ids sometimes lack extension matching
    input_ids = [f"{id}.png" if not id.endswith('.png') else id for id in input_ids]
    
    plot_reconstruction_grid(
        input_ids, 
        input_dir, 
        preds, 
        "reconstruction_emojis.png", 
        "OOD Emoji"
    )

def main():
    # 1. Train the Ridge Probe on the 10,000 dataset (using 80% Train Split)
    print("Loading 10k dataset to train linear probe...")
    ids_pre, X_layers_pre = load_dino_features("dino_pretrained_10k.npz")
    X_pre = X_layers_pre[:, -1, :]
    y = np.load("Z_10k_one_face.npy")
    
    # Split into 80% Train, 20% Test. (Grab the labels to guarantee test samples are completely unseen)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_pre, y, ids_pre, test_size=0.2, random_state=SEED
    )
    
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    print("Linear probe training complete.")
    
    # 2. Reconstruct In-Distribution Images (Random 5 from the completely unseen Test Split)
    reconstruct_unseen_in_distribution(probe, X_test, ids_test)
    
    # 3. Reconstruct Out-Of-Distribution Emojis
    reconstruct_emojis(probe)

if __name__ == "__main__":
    main()
