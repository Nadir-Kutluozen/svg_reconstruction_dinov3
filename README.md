# DINOv3 SVG Extraction & Linear Probing

This repository contains tools to generate parametric SVG face datasets (both single faces and two-face interactions) and evaluate how well pretrained DINOv3 models understand these geometric structures through linear probing.

## Project Structure
- `src/config.py`: Centralized configuration, labels, and file paths.
- `src/dataset/`: Logic for generating procedural SVG face images and their ground-truth latent matrices (`Z`).
- `src/features/`: Scripts to extract `[CLS]` token activations from intermediate layers of DINOv3 models.
- `data/`: The default directory where all generated SVGs, `.npy` targets, and `.npz` DINO features are stored. **(This folder is automatically created when you run the generation scripts).**

## Setup
Ensure you have the following installed by running:
```bash
pip install -r requirements.txt
```

### Hugging Face Authentication
The DINOv3 model requires you to be authenticated with Hugging Face to download the pretrained weights. 
You must generate an Access Token from your Hugging Face account settings. 

Create a file named `.env` in the root of the repository and add your token to it like this:
```bash
HF_TOKEN=your_huggingface_token_here
```
*(The `.gitignore` file already prevents this file from being uploaded to GitHub, keeping your token safe!)*

## How to Run (From Scratch)

If you have just cloned this repository and do not have any datasets yet, follow these steps. The scripts will automatically build the necessary directories inside a `data/` folder.

### Step 1: Generate Datasets
You can generate two different types of datasets: the **Face Dataset** or the **Scene World Dataset**.

**A. Generate the Face Dataset**
Generates simple SVG faces. You can specify the number of faces per image (`--faces 1` or `--faces 2`).
```bash
python main.py generate --faces 1 --samples 10000 --convert
```
* `--faces`: Number of faces (1 or 2).
* `--samples`: Total number of SVGs to generate (default 10000).
* `--convert`: Automatically converts the generated SVGs into PNGs using CairoSVG.

**B. Generate the Scene World Dataset**
Generates the complex Scene World datasets. A single shared `Z_10k_scene.npy` is created to ensure cross-domain probing works perfectly between themes.
```bash
python main.py generate --scene island --samples 10000 --convert
python main.py generate --scene western --samples 10000 --convert
```
* `--scene`: The theme of the scene world (`island` or `western`).
* *(Note: `--faces` and `--scene` are mutually exclusive).*

### Step 2: Extract DINOv3 Features
Extract features from the generated PNGs. This will automatically extract features for BOTH the Pretrained and Randomly Initialized DINOv3 models.

**A. Extract Face Features**
Extracts the `[CLS]` token across all 12 layers.
```bash
python main.py extract --faces 1
```
*Note: This will output `dino_pretrained_10k.npz` and `dino_random_10k.npz` directly into the `data/` directory.*

**B. Extract Scene World Features**
Extracts both the `[CLS]` token and all 196 dense `[PATCH]` tokens from the last layer (required for Scene Decoding).
```bash
python main.py extract --scene island
python main.py extract --scene western
```
*Note: This will output `cls_pre.npy`, `patches_pre.npy`, and `ids_pre.npy` (and their `_rand` equivalents) inside `data/scene_island/features/`.*

### Step 3: Linear Probing Experiments
All linear probing and visualization functions are unified under the `main.py probe` command. **All generated plots will automatically be saved into the `output_pngs/` directory to keep your workspace clean.**

#### Section A: Face Dataset Probing & Figures
These commands execute on the datasets generated via `--faces 1` or `--faces 2`.

* **Standard $R^2$ Bar Chart:**
  `python main.py probe --experiment standard --faces 1`
  *(Generates `variance_explained_comparison.png`)*

* **Predicted vs Ground-Truth Scatter Plots:**
  `python main.py probe --experiment scatter --faces 1`
  *(Generates `predicted_vs_actual_scatter.png`)*

* **Feature Reconstructions:**
  `python main.py probe --experiment reconstruct`
  *(Generates `reconstruction_grid.png`)*

* **3D Variance Explained Surface:**
  `python main.py probe --experiment 3d --faces 1`
  *(Generates `variance_explained_3d_surface.png`)*

* **Out-Of-Distribution / Generalization Evaluation:**
  `python main.py probe --experiment generalization`
  *(Generates `generalization_bars.png`)*

* **Summary / Feature Isolation:**
  `python main.py probe --experiment summary --faces 1`
  *(Generates `summary_4bar.png`)*

* **Layer-by-Layer Progression:**
  `python main.py probe --experiment all_layers --faces 1`
  *(Generates `all_layers_pearson_comparison.png`)*

* **Latent Correlation Matrix:**
  `python main.py probe --experiment correlation --faces 1`
  *(Generates `correlation_matrix.png`)*

* **Fig 1: Face Latents 3D Cube:**
  `python main.py probe --experiment face_fig1`
  *(Generates `fig1_face_latents.png`)*

* **Fig 4a: Nonlinear Representation Plane:**
  `python main.py probe --experiment face_fig4a`
  *(Generates `fig4a_reverse.png`)*

* **Fig 9: Out-of-Distribution Generalization:**
  `python main.py probe --experiment face_ood_fig`
  *(Generates `fig_ood_cross_domain.png`)*

#### Section B: Scene World Probing & Figures
These commands execute the mathematical Ridge Regression and PCA sweeps on the datasets generated via `--scene island` and `--scene western`. By default, they use the `--features cls` mode, but you can append `--features patches_mean` or `--features patches_concat` to test other representations.

* **Scene Forward Probing (Math Only):**
  `python main.py probe --experiment scene_forward --features cls`
  *(Calculates and prints within-domain and cross-domain decoding $R^2$ scores directly to terminal)*

* **Scene Reverse Probing (Math Only):**
  `python main.py probe --experiment scene_reverse --features cls`
  *(Calculates and prints raw and PCA-whitened Reverse $R^2$ scores directly to terminal)*

* **Fig 6 & 7: Scene Forward Decoding Charts:**
  `python main.py probe --experiment scene_forward_plot --features cls`
  *(Dynamically executes Ridge Regressions across both domains, generating `fig_forward_decoding_cls.png`)*

* **Fig 8: Reverse Probing Whitening Curves:**
  `python main.py probe --experiment scene_reverse_plot`
  *(Dynamically computes PCA reductions for $k \in \{32, 64, 128, 256, 768\}$ across `cls` and `patches_mean` features, generating `fig_reverse_whitening.png`)*

* **Canonical Scene World Latents Visualizer:**
  `python main.py probe --experiment scene_world_latents`
  *(Renders the two default themes with directional arrows showing the 32 latent vectors, generating `scene_world_latents.png`)*
