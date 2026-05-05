import numpy as np
from sklearn.linear_model import Ridge

def var_exp(y_true, y_pred):
    """
    Computes Variance Explained (R^2) per output dimension (column-wise).
    Handles array structures safely.
    """
    tot = y_true.var(axis=0) + 1e-6
    res = (y_true - y_pred).var(axis=0)
    return 1.0 - (res / tot)

def fit_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Fits a Ridge regression probe and returns per-dimension R^2 values.
    Returns:
        r2_scores: Array of shape (n_outputs,)
    """
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    preds = probe.predict(X_test)
    return var_exp(y_test, preds)

def load_dino_features(file_path):
    """
    Loads saved DINOv3 features from an .npz file
    Returns:
        image_ids: List of image string IDs
        X_layers: NumPy array of layer features
    """
    print(f"Loading features from {file_path}...")
    loaded = np.load(file_path)
    X_layers = loaded['X_layers']
    all_ids = loaded['all_ids']
    
    # Standardize format back to list of strings
    if all_ids.ndim == 1:
        all_ids = all_ids.tolist()
        
    print(f"Loaded {len(all_ids)} images. Feature matrix shape: {X_layers.shape}")
    return all_ids, X_layers
