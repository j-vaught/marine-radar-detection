"""Run all classical detection methods and save results in COCO format."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classical.cfar import detect_cfar
from src.classical.morphological import detect_morphological
from src.classical.wavelet import detect_wavelet
from src.classical.otsu_ssotsu import detect_otsu, detect_ssotsu
from src.evaluation.metrics import format_predictions_coco, xyxy_to_xywh


def parse_args():
    parser = argparse.ArgumentParser(description="Run classical detection methods")
    parser.add_argument("--ann-path", type=str, required=True,
                        help="Path to COCO annotation JSON (test split)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing tile images")
    parser.add_argument("--config-dir", type=str, default="configs/classical",
                        help="Directory containing classical config YAMLs")
    parser.add_argument("--output-dir", type=str, default="results/classical",
                        help="Output directory for predictions")
    parser.add_argument("--methods", nargs="+",
                        default=["ca_cfar", "go_cfar", "so_cfar", "os_cfar",
                                 "morphological", "wavelet", "otsu", "ssotsu"],
                        help="Methods to run")
    return parser.parse_args()


def run_method(method_name, image, configs):
    """Dispatch to the appropriate detection method."""
    if method_name == "ca_cfar":
        cfg = configs["cfar"]["ca_cfar"]
        bboxes, _ = detect_cfar(image, tuple(cfg["guard_cells"]), tuple(cfg["reference_cells"]),
                                cfg["pfa"], mode="CA", min_area=cfg["min_area"])
    elif method_name == "go_cfar":
        cfg = configs["cfar"]["go_cfar"]
        bboxes, _ = detect_cfar(image, tuple(cfg["guard_cells"]), tuple(cfg["reference_cells"]),
                                cfg["pfa"], mode="GO", min_area=cfg["min_area"])
    elif method_name == "so_cfar":
        cfg = configs["cfar"]["so_cfar"]
        bboxes, _ = detect_cfar(image, tuple(cfg["guard_cells"]), tuple(cfg["reference_cells"]),
                                cfg["pfa"], mode="SO", min_area=cfg["min_area"])
    elif method_name == "os_cfar":
        cfg = configs["cfar"]["os_cfar"]
        bboxes, _ = detect_cfar(image, tuple(cfg["guard_cells"]), tuple(cfg["reference_cells"]),
                                cfg["pfa"], mode="OS", k_os=cfg["k"], min_area=cfg["min_area"])
    elif method_name == "morphological":
        cfg = configs["morphological"]["morphological"]
        bboxes, _, _ = detect_morphological(
            image,
            opening_kernel_size=cfg["opening_kernel_size"],
            opening_iterations=cfg["opening_iterations"],
            closing_kernel_size=cfg["closing_kernel_size"],
            closing_iterations=cfg["closing_iterations"],
            min_area=cfg["min_area"],
            max_area=cfg["max_area"],
        )
    elif method_name == "wavelet":
        cfg = configs["wavelet"]["wavelet"]
        bboxes, _ = detect_wavelet(
            image,
            wavelet=cfg["wavelet"],
            level=cfg["level"],
            threshold_sigma=cfg["threshold_sigma"],
            subbands=cfg["subbands"],
            combine_mode=cfg["combine_mode"],
            min_area=cfg["min_area"],
        )
    elif method_name == "otsu":
        cfg = configs["otsu"]["otsu"]
        bboxes, _ = detect_otsu(image, min_area=cfg["min_area"], max_area=cfg["max_area"])
    elif method_name == "ssotsu":
        cfg = configs["otsu"]["ssotsu"]
        bboxes, _ = detect_ssotsu(
            image,
            block_size=cfg["block_size"],
            seed_threshold_factor=cfg["seed_threshold_factor"],
            region_grow_threshold=cfg["region_grow_threshold"],
            min_area=cfg["min_area"],
            max_area=cfg["max_area"],
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return bboxes


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(args.ann_path) as f:
        ann_data = json.load(f)

    # Load configs
    config_dir = Path(args.config_dir)
    configs = {}
    for cfg_file in config_dir.glob("*.yaml"):
        with open(cfg_file) as f:
            configs[cfg_file.stem] = yaml.safe_load(f)

    # Run each method
    for method_name in args.methods:
        print(f"\nRunning {method_name}...")
        all_predictions = []

        for img_info in tqdm(ann_data["images"], desc=method_name):
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image_float = image.astype(np.float32) / 255.0
            bboxes = run_method(method_name, image_float, configs)

            # Classical methods have no confidence score, use 1.0
            scores = np.ones(len(bboxes), dtype=np.float32)
            preds = format_predictions_coco(img_info["id"], bboxes, scores)
            all_predictions.extend(preds)

        # Save predictions
        pred_path = output_dir / f"{method_name}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(all_predictions, f)
        print(f"  Saved {len(all_predictions)} predictions to {pred_path}")


if __name__ == "__main__":
    main()
