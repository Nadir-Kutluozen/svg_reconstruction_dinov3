import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="DINOv3 SVG Linear Probing Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Generate Dataset
    parser_gen = subparsers.add_parser("generate", help="Generate SVG dataset")
    parser_gen.add_argument("--faces", type=int, choices=[1, 2], default=1, help="Number of faces")
    parser_gen.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser_gen.add_argument("--convert", action="store_true", help="Convert SVG to PNG")

    # 2. Extract Features
    parser_ext = subparsers.add_parser("extract", help="Extract DINOv3 features")
    parser_ext.add_argument("--faces", type=int, choices=[1, 2], default=1)

    # 3. Probe Experiments
    parser_probe = subparsers.add_parser("probe", help="Run linear probing experiments")
    parser_probe.add_argument("--experiment", type=str, choices=["standard", "scatter", "reconstruct", "3d", "generalization", "correlation", "all_layers", "summary"], required=True)
    parser_probe.add_argument("--faces", type=int, choices=[1, 2], default=1)

    args = parser.parse_args()

    if args.command == "generate":
        from src.dataset.generator import build_dataset
        from src.dataset.utils import convert_svgs_to_pngs
        from src.config import DATA_DIR, ONE_FACE_DIR, TWO_FACES_DIR
        import os
        
        build_dataset(num_faces=args.faces, n_total=args.samples, out_dir=DATA_DIR)
        if args.convert:
            base_dir = ONE_FACE_DIR if args.faces == 1 else TWO_FACES_DIR
            convert_svgs_to_pngs(os.path.join(base_dir, "svgs"), os.path.join(base_dir, "pngs"))

    elif args.command == "extract":
        import subprocess
        # Simply call the extractor module since it's cleaner to handle imports
        cmd = [sys.executable, "src/features/extractor.py", "--faces", str(args.faces)]
        subprocess.run(cmd)

    elif args.command == "probe":
        import src.probing.experiments as exp
        
        if args.experiment == "standard":
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
