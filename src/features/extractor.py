import os
import argparse
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader

class PNGDataset(Dataset):
    """Custom PyTorch Dataset for loading and processing images for DINO."""
    def __init__(self, png_dir, png_files, processor):
        self.png_dir = png_dir
        self.png_files = png_files
        self.processor = processor

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        filename = self.png_files[idx]
        image_id = filename.replace(".png", "")
        img_path = os.path.join(self.png_dir, filename)
        
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        return image_id, pixel_values

def extract_dino_features(png_dir, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", pretrained=True, device=None, batch_size=128, extract_patches=False):
    """
    Loads DINOv3 model and extracts intermediate layer [CLS] tokens for all images in the directory using batched inference.
    If extract_patches=True, only extracts the last layer [CLS] and all patch tokens.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor and model ({'pretrained' if pretrained else 'random'})...")
    processor = AutoImageProcessor.from_pretrained(model_id)

    if pretrained:
        model = AutoModel.from_pretrained(model_id)
    else:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_config(config)

    model = model.to(device)
    model.eval()

    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
    if not png_files:
        print(f"No PNG files found in {png_dir}!")
        return [], np.array([]), np.array([])
        
    print(f"Found {len(png_files)} PNG files in {png_dir}. Starting batched extraction...")

    dataset = PNGDataset(png_dir, png_files, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_ids = []
    all_layer_cls = []
    all_patches = []

    with torch.no_grad():
        for batch_ids, batch_pixel_values in tqdm(dataloader, desc="Extracting Batches"):
            batch_pixel_values = batch_pixel_values.to(device)
            outputs = model(pixel_values=batch_pixel_values, output_hidden_states=True)

            if extract_patches:
                n_reg = getattr(model.config, "num_register_tokens", 0)
                hidden = outputs.last_hidden_state
                cls = hidden[:, 0, :].cpu().numpy()
                patches = hidden[:, 1 + n_reg:, :].cpu().numpy()
                all_layer_cls.append(cls)
                all_patches.append(patches)
            else:
                batch_layer_cls = []
                for h in outputs.hidden_states:
                    cls = h[:, 0, :].cpu().numpy()
                    batch_layer_cls.append(cls)
                batch_layer_cls = np.stack(batch_layer_cls, axis=1)
                all_layer_cls.append(batch_layer_cls)
            
            all_ids.extend(batch_ids)

    X_cls = np.concatenate(all_layer_cls, axis=0)
    X_patches = np.concatenate(all_patches, axis=0) if extract_patches else np.array([])

    print("Extraction complete.")
    print("Number of images:", len(all_ids))
    print("CLS shape:", X_cls.shape)
    if extract_patches:
        print("Patches shape:", X_patches.shape)
    
    return all_ids, X_cls, X_patches

def save_dino_features(file_path, image_ids, X_layers):
    print(f"Saving features to {file_path}...")
    np.savez(file_path, all_ids=image_ids, X_layers=X_layers)
    print("Features saved successfully.")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR

    parser = argparse.ArgumentParser(description="Extract DINOv3 Features")
    parser.add_argument("--faces", type=int, choices=[1, 2], default=None, help="Which dataset to process")
    parser.add_argument("--scene", type=str, choices=["island", "western"], default=None, help="Scene world theme")
    parser.add_argument("--png_dir", type=str, default=None, help="Override png directory path")
    parser.add_argument("--out_dir", type=str, default=DATA_DIR, help="Where to save the npz/npy files")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.scene:
        png_dir = os.path.join(args.out_dir, f"scene_{args.scene}", "pngs")
        out_feature_dir = os.path.join(args.out_dir, f"scene_{args.scene}", "features")
        os.makedirs(out_feature_dir, exist_ok=True)
        
        for variant, is_pretrained in [("pre", True), ("rand", False)]:
            print(f"\n--- Extracting {variant} DINOv3 Features for Scene: {args.scene} ---")
            ids, cls_feat, patches_feat = extract_dino_features(png_dir, pretrained=is_pretrained, extract_patches=True)
            if len(ids) > 0:
                np.save(os.path.join(out_feature_dir, f"cls_{variant}.npy"), cls_feat)
                np.save(os.path.join(out_feature_dir, f"patches_{variant}.npy"), patches_feat)
                np.save(os.path.join(out_feature_dir, f"ids_{variant}.npy"), ids)
                print(f"Saved to {out_feature_dir}")
                
    else:
        faces = args.faces if args.faces else 1
        if args.png_dir is None:
            dataset_name = "svg_face_dataset_one_face" if faces == 1 else "svg_face_dataset_two_faces"
            png_dir = os.path.join(args.out_dir, dataset_name, "pngs")
        else:
            png_dir = args.png_dir

        if not os.path.exists(png_dir):
            print(f"Error: Directory {png_dir} does not exist.")
            exit(1)

        suffix = "" if faces == 1 else "_twofaces"
        pre_out = os.path.join(args.out_dir, f"dino_pretrained_10k{suffix}.npz")
        rand_out = os.path.join(args.out_dir, f"dino_random_10k{suffix}.npz")

        # Pretrained Model
        print(f"\n--- Extracting Pretrained DINOv3 Features for {faces}-Face dataset ---")
        ids_pretrained, features_pretrained, _ = extract_dino_features(png_dir, pretrained=True)
        if len(ids_pretrained) > 0:
            save_dino_features(pre_out, ids_pretrained, features_pretrained)
        
        # Random Init Model
        print(f"\n--- Extracting Random DINOv3 Features for {faces}-Face dataset ---")
        ids_random, features_random, _ = extract_dino_features(png_dir, pretrained=False)
        if len(ids_random) > 0:
            save_dino_features(rand_out, ids_random, features_random)
            
    print("All extractions completed successfully!")
