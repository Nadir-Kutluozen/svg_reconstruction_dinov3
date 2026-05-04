import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linear_probe_functions import load_dino_features

def main():
    print("Loading datasets...")
    # 1. Load Pretrained Features
    ids_pre, X_layers_pre = load_dino_features("dino_pretrained_10k.npz")
    
    # 2. Load Random Features
    ids_rand, X_layers_rand = load_dino_features("dino_random_10k.npz")
    
    # 3. Load Target Metadata
    y = np.load("Z_10k_one_face.npy")
    labels = [
        "face radius", "face x position", "face y position", 
        "eye radius", "eye spacing", "eye offset", 
        "mouth width", "mouth offset", "mouth curve",
        "skin H", "skin S", "skin V",
        "eye H", "eye S", "eye V"
    ]
    
    num_layers = X_layers_pre.shape[1]
    num_features = len(labels)
    
    # Matrices to store results: shape (num_layers, num_features)
    pearson_pre = np.zeros((num_layers, num_features))
    pearson_rand = np.zeros((num_layers, num_features))
    
    print(f"Beginning training across all {num_layers} layers. This may take about 30-60 seconds...")
    
    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx + 1}/{num_layers}...")
        
        X_pre = X_layers_pre[:, layer_idx, :]
        X_rand = X_layers_rand[:, layer_idx, :]
        
        # Train/Test Split
        X_train_pre, X_test_pre, y_train, y_test = train_test_split(X_pre, y, test_size=0.2, random_state=42)
        X_train_rand, X_test_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=42)
        
        # Fit Pretrained
        probe_pre = Ridge(alpha=1.0)
        probe_pre.fit(X_train_pre, y_train)
        preds_pre = probe_pre.predict(X_test_pre)
        
        # Fit Random
        probe_rand = Ridge(alpha=1.0)
        probe_rand.fit(X_train_rand, y_train)
        preds_rand = probe_rand.predict(X_test_rand)
        
        # Calculate Pearson correlation for each feature
        for i in range(num_features):
            gt = y_test[:, i]
            
            p_pre = preds_pre[:, i]
            if np.std(p_pre) > 0 and np.std(gt) > 0:
                rho_pre, _ = pearsonr(gt, p_pre)
            else:
                rho_pre = 0.0
            pearson_pre[layer_idx, i] = rho_pre
            
            p_rand = preds_rand[:, i]
            if np.std(p_rand) > 0 and np.std(gt) > 0:
                rho_rand, _ = pearsonr(gt, p_rand)
            else:
                rho_rand = 0.0
            pearson_rand[layer_idx, i] = rho_rand

    # Plotting
    print("Generating plot...")
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
        
        # Put legend in the first plot
        if i == 0:
            ax.legend(loc='lower right', fontsize=10)
            
    # Hide any unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    
    out_file = "all_layers_pearson_comparison.png"
    plt.savefig(out_file, dpi=500, bbox_inches='tight')
    print(f"Saved plot grid to {out_file}!")

if __name__ == "__main__":
    main()
