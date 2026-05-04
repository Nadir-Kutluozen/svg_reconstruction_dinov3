import os
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_probe_functions import load_dino_features

SEED = 42

def var_exp(y, y_):
    """Computes Variance Explained (R^2)"""
    tot_var = y.var(axis=0) + 1e-6
    res_var = (y - y_).var(axis=0)
    return 1 - (res_var / tot_var)

def main():
    print("Loading Pretrained Features...")
    ids_pre, X_layers_pre = load_dino_features("dino_pretrained_10k.npz")
    X_pre = X_layers_pre[:, -1, :]  # Shape: (10000, 768)
    
    print("Loading Random Features...")
    ids_rand, X_layers_rand = load_dino_features("dino_random_10k.npz")
    X_rand = X_layers_rand[:, -1, :]  # Shape: (10000, 768)
    
    print("Loading Latent Ground Truths...")
    y_all = np.load("Z_10k_one_face.npy")  # Shape: (10000, 15)
    
    # Define the variables to test: target_idx -> (target_name, paired_feature_for_compgen)
    # Compgen will test combinations where BOTH target > 0.5 and pair > 0.5
    targets = {
        0:  ("face_radius", 1),  # pair with face_cx (1)
        1:  ("face_cx", 2),      # pair with face_cy (2)
        2:  ("face_cy", 0),      # pair with face_radius (0)
        9:  ("skin_h", 10),      # pair with skin_s (10)
        10: ("skin_s", 11),      # pair with skin_v (11)
        11: ("skin_v", 9)        # pair with skin_h (9)
    }

    modes = ['iid', 'compgen', 'extrapolate']
    
    print("\n--- Running Generalization Tests ---")

    for target_idx, (target_name, pair_idx) in targets.items():
        print(f"\nEvaluating: {target_name.upper()} (Index {target_idx})")
        
        y_target = y_all[:, target_idx:target_idx+1]  # Keep it 2D for Ridge, Shape: (10000, 1)
        
        r2_pre_results = []
        r2_rand_results = []

        for mode in modes:
            if mode == 'iid':
                X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_pre, y_target, test_size=0.2, random_state=SEED)
                X_tr_rand, X_te_rand, _, _ = train_test_split(X_rand, y_target, test_size=0.2, random_state=SEED)
                
            elif mode == 'compgen':
                # Hold out the quadrant where BOTH the target feature AND its paired feature are > 0.5
                ind_test = (y_all[:, target_idx] > 0.5) & (y_all[:, pair_idx] > 0.5)
                ind_train = ~ind_test
                
                X_tr_pre, X_te_pre = X_pre[ind_train], X_pre[ind_test]
                X_tr_rand, X_te_rand = X_rand[ind_train], X_rand[ind_test]
                y_tr, y_te = y_target[ind_train], y_target[ind_test]
                
            elif mode == 'extrapolate':
                # Train on target < 0.5, Test on target >= 0.5
                ind_train = y_all[:, target_idx] < 0.5
                ind_test = ~ind_train
                
                X_tr_pre, X_te_pre = X_pre[ind_train], X_pre[ind_test]
                X_tr_rand, X_te_rand = X_rand[ind_train], X_rand[ind_test]
                y_tr, y_te = y_target[ind_train], y_target[ind_test]
                
            # Fit Pretrained Probe
            probe_pre = Ridge(alpha=1.0)
            probe_pre.fit(X_tr_pre, y_tr)
            preds_pre = probe_pre.predict(X_te_pre)
            r2_pre = var_exp(y_te, preds_pre)[0]  # Take 0th element since it's 1D output
            
            # Fit Random Probe
            probe_rand = Ridge(alpha=1.0)
            probe_rand.fit(X_tr_rand, y_tr)
            preds_rand = probe_rand.predict(X_te_rand)
            r2_rand = var_exp(y_te, preds_rand)[0]
            
            r2_pre_results.append(r2_pre)
            r2_rand_results.append(r2_rand)
            
            print(f"  Mode: {mode:<12} | Pretrained R²: {r2_pre:.4f} | Random R²: {r2_rand:.4f}")

        # Plotting the 3-mode Bar Chart for THIS variable
        fig, ax = plt.subplots(figsize=(4.5, 6))  # Squished even more
        x = np.arange(len(modes))
        width = 0.28  # Made bars thinner
        
        ax.bar(x - width/2, r2_pre_results, width, label='Pretrained', color='royalblue')
        ax.bar(x + width/2, r2_rand_results, width, label='Random', color='tomato')
        
        ax.set_ylabel("Variance Explained ($R^2$)", fontsize=11)
        # Format the title to look nice (e.g. "face_radius" -> "Face Radius")
        nice_name = target_name.replace('_', ' ').title()
        ax.set_title(f"{nice_name}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        # Using '\n' so it fits in a narrower chart
        ax.set_xticklabels(['IID', 'CompGen', 'Extrapolate'], fontsize=11)
        
        # Legend has been removed completely to avoid overlapping with numbers
            
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        
        # Clip the lowest y-value to -0.5 so extreme negative values don't squash the chart
        bottom_y = max(-0.5, min(min(r2_pre_results), min(r2_rand_results), 0) * 1.1)
        ax.set_ylim(bottom_y, 1.1)
        
        # Add values on top of bars
        for i, (pre, rand) in enumerate(zip(r2_pre_results, r2_rand_results)):
            ax.text(i - width/2, max(0, pre) + 0.02, f"{pre:.3f}", ha='center', va='bottom', fontsize=8, fontweight='semibold', rotation=45)
            ax.text(i + width/2, max(0, rand) + 0.02, f"{rand:.3f}", ha='center', va='bottom', fontsize=8, fontweight='semibold', rotation=45)

        # Use strict layout adjustment instead of tight_layout so every chart box is IDENTICAL in size
        fig.subplots_adjust(left=0.2, right=0.95, top=0.90, bottom=0.15)
        plot_path = f"summary_generalization_{target_name}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)  # Close the figure so it doesn't display all 6 if running in an interactive shell
        print(f"  -> Saved chart to: {plot_path}")

if __name__ == "__main__":
    main()
