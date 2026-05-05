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

## How to Run (From Scratch)

If you have just cloned this repository and do not have any datasets yet, follow these steps. The scripts will automatically build the necessary directories inside a `data/` folder.

### Step 1: Generate the SVG Dataset
You can generate the dataset of SVGs along with their ground-truth latent feature matrix (`Z_10k_one_face.npy` or `Z_10k_two_faces.npy`).

To generate 10,000 single-face SVGs and convert them to PNGs:
```bash
python src/dataset/generator.py --faces 1 --samples 10000 --convert
```

To generate 10,000 two-face SVGs and convert them to PNGs:
```bash
python src/dataset/generator.py --faces 2 --samples 10000 --convert
```

### Step 2: Extract DINOv3 Features
Once the PNGs are generated, you can pass them through the DINOv3 model (both pretrained and randomly initialized versions) to extract their layer-wise features.

For the single-face dataset:
```bash
python src/features/extractor.py --faces 1
```

For the two-face dataset:
```bash
python src/features/extractor.py --faces 2
```
*Note: This will output `dino_pretrained_10k.npz` and `dino_random_10k.npz` directly into the `data/` directory.*

### Step 3: Linear Probing Experiments
All linear probing and visualization functions are now united under the `main.py probe` command. **All generated plots will automatically be saved into the `output_pngs/` directory to keep your workspace clean.**

To generate the predicted vs ground-truth scatter plots:
```bash
python main.py probe --experiment scatter --faces 1
```

To generate the 3D Variance Explained surface and the 4-bar standard summary:
```bash
python main.py probe --experiment 3d --faces 1
```

To run a single-face SVG reconstruction from pretrained DINOv3 features:
```bash
python main.py probe --experiment reconstruct
```

To generate the IID, Compositional, and Extrapolation generalization bar charts:
```bash
python main.py probe --experiment generalization
```

To generate the simple Pretrained vs Random $R^2$ bar chart (from the standard probe):
```bash
python main.py probe --experiment standard --faces 1
```

To generate the 4-bar summary comparing Feature Decoding (Activations $\rightarrow$ Features) vs Encoding (Features $\rightarrow$ Activations):
```bash
python main.py probe --experiment summary --faces 1
```

To generate the correlation matrix of the ground-truth generative $Z$ features:
```bash
python main.py probe --experiment correlation --faces 1
```

To generate the layer-by-layer Pearson correlation progression line chart:
```bash
python main.py probe --experiment all_layers --faces 1
```
