import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from PIL import Image
import cairosvg

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANVAS_W, CANVAS_H
from dataset.utils import make_mouth_path

# ==========================================
# 1. SCATTER PLOTS
# ==========================================
def plot_scatter_grid(y_true, y_pred, labels, title, filename):
    """
    Plots a grid of scatter plots (Ground Truth vs Predicted) for each feature.
    Calculates and displays the Pearson correlation coefficient (rho) in the title.
    """
    num_features = len(labels)
    cols = 5
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=.6 * np.array((20, 3.5 * rows)))
    axes = axes.flatten()
    
    for i, label in enumerate(labels):
        ax = axes[i]
        gt = y_true[:, i]
        pred = y_pred[:, i]
        
        if np.std(pred) == 0 or np.std(gt) == 0:
            rho = 0.0
        else:
            rho, _ = pearsonr(gt, pred)
            
        ax.scatter(gt, pred, s=3, alpha=0.8, color='cornflowerblue')
        ax.set_title(f"{label}")
        ax.text(0.05, 0.95, rf"$\rho={rho:.4f}$", transform=ax.transAxes)
        if i % cols == 0:
            ax.set_ylabel("Predicted")
        else:
            ax.set_yticklabels([])
        if i < cols*(rows-1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Ground Truth")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    print(f"Saved scatter plot grid to {filename}")
    plt.close()

# ==========================================
# 2. LAYERWISE PEARSON CORRELATION
# ==========================================
def plot_layerwise_pearson(pearson_pre, pearson_rand, labels, filename):
    """
    Plots layer-by-layer Pearson correlation progression.
    pearson_pre, pearson_rand: Arrays of shape (num_layers, num_features)
    """
    num_layers = pearson_pre.shape[0]
    num_features = len(labels)
    
    cols = 5
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=.6 * np.array((20, 3.5 * rows)))
    axes = axes.flatten()
    
    layers_x = np.arange(1, num_layers + 1)
    
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(layers_x, pearson_pre[:, i], marker='o', linestyle='-', label='Pretrained', color='#1f77b4', linewidth=2)
        ax.plot(layers_x, pearson_rand[:, i], marker='x', linestyle='--', label='Random', color='#d62728')
        
        ax.set_title(f"{label}")
        if i % cols == 0:
            ax.set_ylabel(r"Pearson $\rho$")
        else:
            ax.set_yticklabels([])
            
        if i < cols*(rows-1):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Layer Index")
            
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([-0.1, 1.05])
        
        if i == 0:
            ax.legend(loc='lower right', fontsize=10)
            
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    print(f"Saved layerwise plot grid to {filename}")
    plt.close()

# ==========================================
# 3. 3D SURFACE PLOT
# ==========================================
def plot_3d_surface(surface_pre, surface_rand, labels, save_path="r2_surface_3d.png"):
    """
    Two-panel 3-D surface showing R2 across latent features and activation dims.
    """
    n_feats, n_acts = surface_pre.shape
    act_idx  = np.arange(n_acts)
    feat_idx = np.arange(n_feats)
    ACT, FEAT = np.meshgrid(act_idx, feat_idx)

    z_min = min(surface_pre.min(), surface_rand.min(), 0.0)
    z_max = max(surface_pre.max(), surface_rand.max())

    fig = plt.figure(figsize=(20, 16))
    
    # Pretrained
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(ACT, FEAT, surface_pre, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("Activation dim", fontsize=11, labelpad=8)
    ax1.set_ylabel("Latent feature", fontsize=11, labelpad=12)
    ax1.set_zlabel("Variance Explained ($R^2$)", fontsize=11, labelpad=6)
    ax1.set_yticks(feat_idx)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_title("Pretrained DINOv3", fontsize=18, fontweight="bold", pad=15)
    fig.colorbar(surf1, ax=ax1, shrink=0.45, label="$R^2$", pad=0.15)

    # Random
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(ACT, FEAT, surface_rand, cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("Activation dim", fontsize=11, labelpad=8)
    ax2.set_ylabel("Latent feature", fontsize=11, labelpad=12)
    ax2.set_zlabel("Variance Explained ($R^2$)", fontsize=11, labelpad=6)
    ax2.set_yticks(feat_idx)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title("Random DINOv3", fontsize=18, fontweight="bold", pad=15)
    fig.colorbar(surf2, ax=ax2, shrink=0.45, label="$R^2$", pad=0.15)

    # Difference heatmap
    diff = surface_rand - surface_pre
    ax3 = fig.add_subplot(2, 2, 3)
    vabs = np.abs(diff).max()
    im = ax3.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
    ax3.set_xlabel("Activation dimension", fontsize=10)
    ax3.set_ylabel("Latent feature", fontsize=10)
    ax3.set_yticks(feat_idx)
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_title("Difference  (Random − Pretrained)\nBlue = pretrained wins · Red = random wins", fontsize=11)
    fig.colorbar(im, ax=ax3, label="ΔR²")

    # Bar chart
    ax4 = fig.add_subplot(2, 2, 4)
    avg_pre  = surface_pre.mean(axis=1)
    avg_rand = surface_rand.mean(axis=1)
    x = np.arange(n_feats)
    w = 0.35
    ax4.barh(x - w/2, avg_pre,  w, label="Pretrained", color="royalblue")
    ax4.barh(x + w/2, avg_rand, w, label="Random",     color="tomato")
    ax4.set_yticks(x)
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel("Mean R² across all activation dims", fontsize=9)
    ax4.set_title("Which latent features are most recoverable?", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    plt.close()

# ==========================================
# 4. R2 BAR CHARTS
# ==========================================
def plot_r2_bar_chart(r2_pre, r2_rand, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, r2_pre, width, label='Pretrained', color='royalblue')
    rects2 = ax.bar(x + width/2, r2_rand, width, label='Random', color='tomato')
    
    ax.set_ylabel('Variance Explained ($R^2$)', fontsize=12)
    ax.set_title('Linear Probe DINOv3 Features', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved bar chart to: {save_path}")
    plt.close()

def plot_4bar_summary(avg_std_pre, avg_std_rand, avg_rev_pre, avg_rev_rand, save_path="summary_4bar.png"):
    bars    = [avg_std_pre, avg_std_rand, avg_rev_pre, avg_rev_rand]
    colors  = ["royalblue", "tomato", "royalblue", "tomato"]
    xlabels = ["Pretrained", "Random", "Pretrained", "Random"]

    fig, ax = plt.subplots(figsize=(7, 6))
    x     = np.arange(4)
    rects = ax.bar(x, bars, color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    for rect, val in zip(rects, bars):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylim(0, max(bars) * 1.25)
    
    ax.text(0.5, 1.02, "Feature Decoding\n(Activations → Features)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=12, color="black", fontweight="bold")
    ax.text(2.5, 1.02, "Feature Encoding\n(Features → Activations)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=12, color="black", fontweight="bold")

    ax.legend(handles=[Patch(color="royalblue", label="Pretrained DINOv3"), Patch(color="tomato", label="Random DINOv3")], 
              fontsize=11, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel("Variance Explained ($R^2$)", fontsize=13)
    ax.set_title("Linear Probe Summary (DINOv3)", fontsize=15, fontweight="bold", y=1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()

# ==========================================
# 5. GENERALIZATION BAR CHARTS
# ==========================================
def plot_generalization_bars(target_name, modes, r2_pre_results, r2_rand_results, out_file):
    fig, ax = plt.subplots(figsize=(4.5, 6))
    x = np.arange(len(modes))
    width = 0.28
    
    ax.bar(x - width/2, r2_pre_results, width, label='Pretrained', color='royalblue')
    ax.bar(x + width/2, r2_rand_results, width, label='Random', color='tomato')
    
    ax.set_ylabel("Variance Explained ($R^2$)", fontsize=11)
    nice_name = target_name.replace('_', ' ').title()
    ax.set_title(f"{nice_name}", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['IID', 'CompGen', 'Extrapolate'], fontsize=11)
        
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    
    bottom_y = max(-0.5, min(min(r2_pre_results), min(r2_rand_results), 0) * 1.1)
    ax.set_ylim(bottom_y, 1.1)
    
    for i, (pre, rand) in enumerate(zip(r2_pre_results, r2_rand_results)):
        ax.text(i - width/2, max(0, pre) + 0.02, f"{pre:.3f}", ha='center', va='bottom', fontsize=8, fontweight='semibold', rotation=45)
        ax.text(i + width/2, max(0, rand) + 0.02, f"{rand:.3f}", ha='center', va='bottom', fontsize=8, fontweight='semibold', rotation=45)

    fig.subplots_adjust(left=0.2, right=0.95, top=0.90, bottom=0.15)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved chart to: {out_file}")

# ==========================================
# 6. CORRELATION MATRIX
# ==========================================
def plot_correlation_matrix(feature_matrix, labels, out_file):
    corr = np.corrcoef(feature_matrix, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.colorbar(label="Pearson Correlation")
    
    plt.title("Correlation Matrix of Generative Z Features", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved correlation matrix to: {out_file}")

# ==========================================
# 7. RECONSTRUCTION RENDERING
# ==========================================
def build_svg_from_predictions(preds):
    import colorsys
    z = np.clip(preds, 0.0, 1.0)
    
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
    png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=CANVAS_W, output_height=CANVAS_H)
    return Image.open(io.BytesIO(png_data))

def plot_reconstruction_grid(image_ids, source_dir, predicted_features, output_filename, title_prefix=""):
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
             orig_img_path = os.path.join(source_dir, f"{img_id}")
        
        orig_pil = Image.open(orig_img_path).convert("RGB")
        predicted_svg_str = build_svg_from_predictions(predicted_features[i])
        predicted_pil = render_svg_to_pil(predicted_svg_str)
        
        axes[i, 0].imshow(orig_pil)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        axes[i, 1].imshow(predicted_pil)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        for spine in axes[i, 1].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"Saved side-by-side reconstruction to {output_filename}")
    plt.close()
