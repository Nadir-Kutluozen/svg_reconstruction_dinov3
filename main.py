import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="DINOv3 SVG Linear Probing Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Generate Dataset
    parser_gen = subparsers.add_parser("generate", help="Generate SVG dataset")
    parser_gen.add_argument("--faces", type=int, choices=[1, 2], default=None, help="Number of faces")
    parser_gen.add_argument("--scene", type=str, choices=["island", "western"], default=None, help="Scene world theme")
    parser_gen.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser_gen.add_argument("--convert", action="store_true", help="Convert SVG to PNG")

    # 2. Extract Features
    parser_ext = subparsers.add_parser("extract", help="Extract DINOv3 features")
    parser_ext.add_argument("--faces", type=int, choices=[1, 2], default=None)
    parser_ext.add_argument("--scene", type=str, choices=["island", "western"], default=None)

    # 3. Probe Experiments
    parser_probe = subparsers.add_parser("probe", help="Run linear probing experiments")
    parser_probe.add_argument("--experiment", type=str, choices=[
        "standard", "scatter", "reconstruct", "3d", "generalization", "correlation", "all_layers", "summary",
        "scene_forward", "scene_reverse", "scene_forward_plot", "scene_reverse_plot", "scene_world_latents",
        "face_fig1", "face_fig4a", "face_ood_fig"
    ], required=True)
    parser_probe.add_argument("--faces", type=int, choices=[1, 2], default=1)
    parser_probe.add_argument("--features", type=str, choices=["cls", "patches_mean", "patches_concat"], default="cls", help="Feature mode for scene experiments")

    args = parser.parse_args()

    if args.command == "generate":
        from src.dataset.generator import build_dataset, build_scene_dataset
        from src.dataset.utils import convert_svgs_to_pngs
        from src.config import DATA_DIR, ONE_FACE_DIR, TWO_FACES_DIR
        import os
        
        if args.scene:
            build_scene_dataset(theme=args.scene, n_total=args.samples, out_dir=DATA_DIR)
            if args.convert:
                base_dir = os.path.join(DATA_DIR, f"scene_{args.scene}")
                convert_svgs_to_pngs(os.path.join(base_dir, "svgs"), os.path.join(base_dir, "pngs"))
        else:
            faces = args.faces if args.faces else 1
            build_dataset(num_faces=faces, n_total=args.samples, out_dir=DATA_DIR)
            if args.convert:
                base_dir = ONE_FACE_DIR if faces == 1 else TWO_FACES_DIR
                convert_svgs_to_pngs(os.path.join(base_dir, "svgs"), os.path.join(base_dir, "pngs"))

    elif args.command == "extract":
        import subprocess
        # Simply call the extractor module since it's cleaner to handle imports
        cmd = [sys.executable, "src/features/extractor.py"]
        if args.scene:
            cmd.extend(["--scene", args.scene])
        else:
            faces = args.faces if args.faces else 1
            cmd.extend(["--faces", str(faces)])
        subprocess.run(cmd)

    elif args.command == "probe":
        import src.probing.experiments as exp
        
        if args.experiment == "scene_forward":
            from src.probing.scene_experiments import run_forward_probing, USED_DIMS, LATENT_LABELS
            
            print(f"\n[Scene Forward Probing] Feature Mode: {args.features}")
            for theme in ["island", "western"]:
                for variant in ["pre", "rand"]:
                    r2, rho = run_forward_probing(theme, theme, variant, mode=args.features)
                    r2_mean = r2[USED_DIMS].mean()
                    print(f"Within-Domain ({theme} -> {theme}) | {variant} | R^2 Mean (Used Dims): {r2_mean:.4f}")
            
            print("\n")
            for src, tgt in [("island", "western"), ("western", "island")]:
                for variant in ["pre", "rand"]:
                    r2, rho = run_forward_probing(src, tgt, variant, mode=args.features)
                    r2_mean = r2[USED_DIMS].mean()
                    print(f"Cross-Domain  ({src} -> {tgt}) | {variant} | R^2 Mean (Used Dims): {r2_mean:.4f}")
                    
        elif args.experiment == "scene_reverse":
            from src.probing.scene_experiments import run_reverse_probing
            
            print(f"\n[Scene Reverse Probing / Feature Isolation] Feature Mode: {args.features}")
            for variant in ["pre", "rand"]:
                raw_list = []
                whitened_list = []
                for theme in ["island", "western"]:
                    res = run_reverse_probing(theme, variant, mode=args.features)
                    raw_list.append(res["raw"])
                    whitened_list.append(res["pca_whitened"])
                    
                print(f"Variant: {variant}")
                print(f"  Raw R^2 (Mean over themes):          {sum(raw_list)/2:.4f}")
                print(f"  Whitened R^2 (Mean over themes):     {sum(whitened_list)/2:.4f}")

        elif args.experiment == "scene_forward_plot":
            from src.visualization.scene_plotter import make_forward_figs
            make_forward_figs(feature_mode=args.features)

        elif args.experiment == "scene_reverse_plot":
            from src.visualization.scene_plotter import make_reverse_fig
            make_reverse_fig()

        elif args.experiment == "scene_world_latents":
            from src.visualization.scene_world_fig import make_scene_world_figure
            make_scene_world_figure()

        elif args.experiment == "face_fig1":
            import subprocess
            subprocess.run([sys.executable, "src/visualization/face_fig1.py"])
            
        elif args.experiment == "face_fig4a":
            import subprocess
            subprocess.run([sys.executable, "src/visualization/face_fig4a.py"])
            
        elif args.experiment == "face_ood_fig":
            import subprocess
            subprocess.run([sys.executable, "src/visualization/face_ood_fig.py"])

        elif args.experiment == "standard":
            exp.run_standard_probe(args.faces)
        elif args.experiment == "scatter":
            exp.run_scatter(args.faces)
        elif args.experiment == "reconstruct":
            if args.faces == 2:
                print("Reconstruction is currently optimized for 1 face only.")
            else:
                exp.run_reconstruct()
        elif args.experiment == "3d":
            exp.run_3d_surface(args.faces)
        elif args.experiment == "generalization":
            exp.run_generalization()
        elif args.experiment == "correlation":
            exp.run_correlation(args.faces)
        elif args.experiment == "all_layers":
            exp.run_all_layers(args.faces)
        elif args.experiment == "summary":
            exp.run_summary(args.faces)

if __name__ == "__main__":
    main()
