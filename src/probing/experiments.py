import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OUTPUT_DIR, SEED, ONE_FACE_LABELS, TWO_FACES_LABELS
from probing.utils import var_exp, fit_probe, load_dino_features
import visualization.plotter as plotter

def load_data(faces=1, layer_idx=-1, load_all_layers=False):
    """Loads feature matrices and ground truth based on 1 or 2 faces."""
    if faces == 1:
        z_path = os.path.join(DATA_DIR, "Z_10k_one_face.npy")
        pre_path = os.path.join(DATA_DIR, "dino_pretrained_10k.npz")
        rand_path = os.path.join(DATA_DIR, "dino_random_10k.npz")
        labels = ONE_FACE_LABELS
    else:
        z_path = os.path.join(DATA_DIR, "Z_10k_two_faces.npy")
        pre_path = os.path.join(DATA_DIR, "dino_pretrained_10k_twofaces.npz")
        rand_path = os.path.join(DATA_DIR, "dino_random_10k_twofaces.npz")
        labels = TWO_FACES_LABELS

    y = np.load(z_path)
    ids_pre, X_layers_pre = load_dino_features(pre_path)
    ids_rand, X_layers_rand = load_dino_features(rand_path)
    
    if not load_all_layers:
        X_pre = X_layers_pre[:, layer_idx, :]
        X_rand = X_layers_rand[:, layer_idx, :]
        return X_pre, X_rand, y, ids_pre, labels
    else:
        return X_layers_pre, X_layers_rand, y, ids_pre, labels

def run_standard_probe(faces=1):
    X_pre, X_rand, y, _, labels = load_data(faces)
    X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_pre, y, test_size=0.2, random_state=SEED)
    X_tr_rand, X_te_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)

    r2_pre = fit_probe(X_tr_pre, y_tr, X_te_pre, y_te)
    r2_rand = fit_probe(X_tr_rand, y_tr, X_te_rand, y_te)
    
    print("\nVariance Explained (R^2) per Feature:")
    for i, label in enumerate(labels):
        print(f"{label:<30} | Pretrained: {r2_pre[i]:.4f} | Random: {r2_rand[i]:.4f}")
        
    out_name = "variance_explained_comparison.png" if faces == 1 else "variance_explained_comparison_two_faces.png"
    plotter.plot_r2_bar_chart(r2_pre, r2_rand, labels, os.path.join(OUTPUT_DIR, out_name))
    
    return r2_pre, r2_rand

def run_scatter(faces=1):
    X_pre, X_rand, y, _, labels = load_data(faces)
    X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_pre, y, test_size=0.2, random_state=SEED)
    X_tr_rand, X_te_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)

    probe_pre = Ridge(alpha=1.0)
    probe_pre.fit(X_tr_pre, y_tr)
    preds_pre = probe_pre.predict(X_te_pre)
    plotter.plot_scatter_grid(y_te, preds_pre, labels, "Pretrained", os.path.join(OUTPUT_DIR, "scatter_pretrained.png"))

    probe_rand = Ridge(alpha=1.0)
    probe_rand.fit(X_tr_rand, y_tr)
    preds_rand = probe_rand.predict(X_te_rand)
    plotter.plot_scatter_grid(y_te, preds_rand, labels, "Random", os.path.join(OUTPUT_DIR, "scatter_random.png"))

def run_reconstruct():
    # Only applies to 1-face currently in original logic
    X_pre, X_rand, y, ids_pre, labels = load_data(faces=1)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X_pre, y, ids_pre, test_size=0.2, random_state=SEED)
    
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    
    # Random 5 indices
    indices = np.random.choice(len(X_test), 5, replace=False)
    sample_ids = [ids_test[i] for i in indices]
    sample_X = X_test[indices]
    preds = probe.predict(sample_X)
    
    pngs_dir = os.path.join(DATA_DIR, "svg_face_dataset_one_face", "pngs")
    plotter.plot_reconstruction_grid(sample_ids, pngs_dir, preds, os.path.join(OUTPUT_DIR, "reconstruction_in_distribution.png"), "In-Distribution")

def run_3d_surface(faces=1):
    X_pre, X_rand, y, _, labels = load_data(faces)
    y_tr, y_te, X_tr_pre, X_te_pre = train_test_split(y, X_pre, test_size=0.2, random_state=SEED)
    _, _, X_tr_rand, X_te_rand = train_test_split(y, X_rand, test_size=0.2, random_state=SEED)

    n_features = y_tr.shape[1]
    n_acts = X_tr_pre.shape[1]
    surface_pre = np.zeros((n_features, n_acts))
    surface_rand = np.zeros((n_features, n_acts))

    print("Fitting per-feature probes (this takes ~30s)...")
    for fi in range(n_features):
        surface_pre[fi] = fit_probe(y_tr[:, fi:fi+1], X_tr_pre, y_te[:, fi:fi+1], X_te_pre)
        surface_rand[fi] = fit_probe(y_tr[:, fi:fi+1], X_tr_rand, y_te[:, fi:fi+1], X_te_rand)

    np.savez(os.path.join(DATA_DIR, "r2_surfaces_z_15.npz"), surface_pre=surface_pre, surface_rand=surface_rand, labels=labels)
    plotter.plot_3d_surface(surface_pre, surface_rand, labels, os.path.join(OUTPUT_DIR, "r2_surface_3d.png"))

def run_generalization():
    X_pre, X_rand, y_all, _, _ = load_data(faces=1)
    targets = {
        0:  ("face_radius", 1), 1:  ("face_cx", 2), 2:  ("face_cy", 0),
        9:  ("skin_h", 10), 10: ("skin_s", 11), 11: ("skin_v", 9)
    }
    modes = ['iid', 'compgen', 'extrapolate']

    for target_idx, (target_name, pair_idx) in targets.items():
        y_target = y_all[:, target_idx:target_idx+1]
        r2_pre_results, r2_rand_results = [], []

        for mode in modes:
            if mode == 'iid':
                X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_pre, y_target, test_size=0.2, random_state=SEED)
                X_tr_rand, X_te_rand, _, _ = train_test_split(X_rand, y_target, test_size=0.2, random_state=SEED)
            elif mode == 'compgen':
                ind_test = (y_all[:, target_idx] > 0.5) & (y_all[:, pair_idx] > 0.5)
                ind_train = ~ind_test
                X_tr_pre, X_te_pre = X_pre[ind_train], X_pre[ind_test]
                X_tr_rand, X_te_rand = X_rand[ind_train], X_rand[ind_test]
                y_tr, y_te = y_target[ind_train], y_target[ind_test]
            elif mode == 'extrapolate':
                ind_train = y_all[:, target_idx] < 0.5
                ind_test = ~ind_train
                X_tr_pre, X_te_pre = X_pre[ind_train], X_pre[ind_test]
                X_tr_rand, X_te_rand = X_rand[ind_train], X_rand[ind_test]
                y_tr, y_te = y_target[ind_train], y_target[ind_test]

            probe_pre = Ridge(alpha=1.0)
            probe_pre.fit(X_tr_pre, y_tr)
            r2_pre_results.append(var_exp(y_te, probe_pre.predict(X_te_pre))[0])

            probe_rand = Ridge(alpha=1.0)
            probe_rand.fit(X_tr_rand, y_tr)
            r2_rand_results.append(var_exp(y_te, probe_rand.predict(X_te_rand))[0])

        out_file = os.path.join(OUTPUT_DIR, f"summary_generalization_{target_name}.png")
        plotter.plot_generalization_bars(target_name, modes, r2_pre_results, r2_rand_results, out_file)

def run_correlation(faces=1):
    X_pre, X_rand, y, _, labels = load_data(faces)
    plotter.plot_correlation_matrix(y, labels, os.path.join(OUTPUT_DIR, "correlation_matrix.png"))

def run_all_layers(faces=1):
    X_layers_pre, X_layers_rand, y, _, labels = load_data(faces, load_all_layers=True)
    num_layers = X_layers_pre.shape[1]
    
    pearson_pre = np.zeros((num_layers, len(labels)))
    pearson_rand = np.zeros((num_layers, len(labels)))
    
    from scipy.stats import pearsonr
    print(f"Fitting probes for all {num_layers} layers...")
    for l in range(num_layers):
        X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_layers_pre[:, l, :], y, test_size=0.2, random_state=SEED)
        probe = Ridge(alpha=1.0)
        probe.fit(X_tr_pre, y_tr)
        preds = probe.predict(X_te_pre)
        for f in range(len(labels)):
            pearson_pre[l, f], _ = pearsonr(y_te[:, f], preds[:, f])
            
        X_tr_rand, X_te_rand, _, _ = train_test_split(X_layers_rand[:, l, :], y, test_size=0.2, random_state=SEED)
        probe_rand = Ridge(alpha=1.0)
        probe_rand.fit(X_tr_rand, y_tr)
        preds_rand = probe_rand.predict(X_te_rand)
        for f in range(len(labels)):
            pearson_rand[l, f], _ = pearsonr(y_te[:, f], preds_rand[:, f])
            
    plotter.plot_layerwise_pearson(pearson_pre, pearson_rand, labels, os.path.join(OUTPUT_DIR, "all_layers_pearson_comparison.png"))

def run_summary(faces=1):
    X_pre, X_rand, y, _, labels = load_data(faces)
    X_tr_pre, X_te_pre, y_tr, y_te = train_test_split(X_pre, y, test_size=0.2, random_state=SEED)
    X_tr_rand, X_te_rand, _, _ = train_test_split(X_rand, y, test_size=0.2, random_state=SEED)

    # Standard (Activations -> Features)
    r2_std_pre = fit_probe(X_tr_pre, y_tr, X_te_pre, y_te)
    r2_std_rand = fit_probe(X_tr_rand, y_tr, X_te_rand, y_te)
    
    # Reverse (Features -> Activations)
    r2_rev_pre = fit_probe(y_tr, X_tr_pre, y_te, X_te_pre)
    r2_rev_rand = fit_probe(y_tr, X_tr_rand, y_te, X_te_rand)
    
    avg_std_pre = np.mean(np.maximum(0, r2_std_pre))
    avg_std_rand = np.mean(np.maximum(0, r2_std_rand))
    avg_rev_pre = np.mean(np.maximum(0, r2_rev_pre))
    avg_rev_rand = np.mean(np.maximum(0, r2_rev_rand))
    
    out_name = "summary_4bar.png" if faces == 1 else "summary_4bar_two_faces.png"
    plotter.plot_4bar_summary(avg_std_pre, avg_std_rand, avg_rev_pre, avg_rev_rand, os.path.join(OUTPUT_DIR, out_name))
