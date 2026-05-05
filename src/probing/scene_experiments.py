import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.config import DATA_DIR, SEED

# ----------------------------------------------------------------------------
# Latent layout (must match svg_island.py / svg_western.py)
# ----------------------------------------------------------------------------
def _make_labels():
    labels = []
    per_person = ["x", "z", "facing", "skin_h", "shirt_h_un", "shirt_s_un", "shirt_v_un"]
    for p in range(4):
        for d in per_person:
            labels.append(f"p{p}_{d}")
    labels += ["dolphin_x", "dolphin_z", "turtle_x", "turtle_z"]
    return labels

LATENT_LABELS = _make_labels()                          # length 32

# 20 dims actually consumed by the renderer.
USED_DIMS = (
    [p * 7 + i for p in range(4) for i in (0, 1, 2, 3)]  # x, z, facing, skin_h
    + [28, 29, 30, 31]                                    # animal positions
)
USED_LABELS = [LATENT_LABELS[i] for i in USED_DIMS]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def r2_per_dim(y_true, y_pred):
    """1 - SSE/SST per output column. Equivalent to sklearn r2_score with
    multioutput='raw_values' but with a small variance floor for stability."""
    tot = y_true.var(axis=0) + 1e-8
    res = (y_true - y_pred).var(axis=0)
    return 1.0 - res / tot

def pearson_per_dim(y_true, y_pred):
    """Pearson correlation per output column. Insensitive to bias and scale,
    so a probe that has the right direction but wrong intercept/slope (e.g.
    cross-domain) shows high ρ even when R² is negative."""
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12
    return num / den

def load_scene_features(theme, variant, mode="cls", n_pca=1024, fit_pca_on=None):
    """Returns an (N, d) feature matrix according to `mode`.

    For mode == 'patches_concat' a PCA fitted on `fit_pca_on` (or self,
    if None) is applied. fit_pca_on is the matrix used for fitting; that
    lets cross-domain probing fit PCA on the source theme and reuse it.
    Returns (X, pca_object_or_None).
    """
    fdir = os.path.join(DATA_DIR, f"scene_{theme}", "features")
    if mode == "cls":
        return np.load(os.path.join(fdir, f"cls_{variant}.npy")), None

    patches = np.load(os.path.join(fdir, f"patches_{variant}.npy"))   # (N, P, D)
    if mode == "patches_mean":
        return patches.mean(axis=1), None
    if mode == "patches_concat":
        flat = patches.reshape(patches.shape[0], -1)                  # (N, P*D)
        if fit_pca_on is None:
            max_k = min(flat.shape)
            if max_k > 0: max_k -= 1
            if n_pca > max_k: n_pca = max_k
            if n_pca <= 0:
                # Fallback if extremely small dataset
                return flat.astype(np.float32), None
                
            pca = PCA(n_components=n_pca, random_state=SEED).fit(flat)
            return pca.transform(flat).astype(np.float32), pca
        # reuse a previously-fit PCA
        return fit_pca_on.transform(flat).astype(np.float32), fit_pca_on
    raise ValueError(f"unknown features mode: {mode}")

def load_scene_latents():
    return np.load(os.path.join(DATA_DIR, "Z_10k_scene.npy"))

# ----------------------------------------------------------------------------
# Forward Probing (Decoding)
# ----------------------------------------------------------------------------
def fit_eval_forward(X_train, X_test, y_train, y_test, alpha):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    return r2_per_dim(y_test, y_pred), pearson_per_dim(y_test, y_pred)

def run_forward_probing(theme_train, theme_test, variant, mode="cls", alpha=1.0, n_pca=1024):
    """
    If theme_train == theme_test: within-domain probing.
    If theme_train != theme_test: cross-domain probing (with per-domain mean centering).
    """
    Z = load_scene_latents()
    
    if theme_train == theme_test:
        X, _ = load_scene_features(theme_train, variant, mode, n_pca)
        X_tr, X_te, y_tr, y_te = train_test_split(X, Z, test_size=0.5, random_state=SEED)
        return fit_eval_forward(X_tr, X_te, y_tr, y_te, alpha)
    else:
        # Cross domain
        X_src, pca = load_scene_features(theme_train, variant, mode, n_pca)
        X_tgt, _   = load_scene_features(theme_test, variant, mode, n_pca, fit_pca_on=pca)

        # Split identically since Z is identical across themes
        X_src_tr, _,         y_tr,   _    = train_test_split(X_src, Z, test_size=0.5, random_state=SEED)
        _,        X_tgt_te,  _,      y_te = train_test_split(X_tgt, Z, test_size=0.5, random_state=SEED)

        # Per-domain centering isolates direction transfer, ignoring bias transfer.
        mu_src = X_src_tr.mean(axis=0, keepdims=True)
        mu_tgt = X_tgt_te.mean(axis=0, keepdims=True)
        X_src_tr = X_src_tr - mu_src
        X_tgt_te = X_tgt_te - mu_tgt

        return fit_eval_forward(X_src_tr, X_tgt_te, y_tr, y_te, alpha)


# ----------------------------------------------------------------------------
# Reverse Probing (Encoding / Feature Isolation)
# ----------------------------------------------------------------------------
def reverse_r2(Z_train, X_train, Z_test, X_test, alpha=1.0):
    """Single scalar reverse R^2: fraction of total feature variance explained
    by a linear function of latents. Computed on test set."""
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(Z_train, X_train)
    X_pred = probe.predict(Z_test)
    total = X_test.var(axis=0).sum() + 1e-12
    resid = (X_test - X_pred).var(axis=0).sum()
    return 1.0 - resid / total

def standardize(X_train, X_test):
    """Z-score per dim, fit stats on train."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd

def pca_whiten(X_train, X_test, n_components=None):
    """PCA-whiten. Caps n_components to min(N_train, D) - 1 safely."""
    max_k = min(X_train.shape)
    if max_k > 0:
        max_k -= 1
        
    if n_components is None or n_components > max_k:
        n_components = max_k
        
    if n_components <= 0:
        # If dataset is too small, just return it without PCA to avoid crashing
        return X_train, X_test
        
    pca = PCA(n_components=n_components, whiten=True, random_state=SEED)
    return pca.fit_transform(X_train), pca.transform(X_test)

def run_reverse_probing(theme, variant, mode="cls", alpha=1.0, n_pca=1024, n_pca_whiten=None):
    """
    Runs reverse probing Z -> f(x) and computes R^2 under three normalizations:
    raw, standardized, and PCA-whitened.
    Returns: dict of { "raw": r2, "standardized": r2, "pca_whitened": r2, "whiten_k": k }
    """
    Z = load_scene_latents()
    X, _ = load_scene_features(theme, variant, mode, n_pca)
    
    X_tr, X_te, Z_tr, Z_te = train_test_split(X, Z, test_size=0.5, random_state=SEED)

    # Raw
    r2_raw = reverse_r2(Z_tr, X_tr, Z_te, X_te, alpha=alpha)

    # Standardized
    X_tr_s, X_te_s = standardize(X_tr, X_te)
    r2_std = reverse_r2(Z_tr, X_tr_s, Z_te, X_te_s, alpha=alpha)

    # PCA-whitened
    n_w = n_pca_whiten or (min(X_tr.shape) - 1)
    X_tr_w, X_te_w = pca_whiten(X_tr, X_te, n_components=n_w)
    r2_pw = reverse_r2(Z_tr, X_tr_w, Z_te, X_te_w, alpha=alpha)

    return {
        "raw": float(r2_raw),
        "standardized": float(r2_std),
        "pca_whitened": float(r2_pw),
        "whiten_k": int(n_w)
    }
