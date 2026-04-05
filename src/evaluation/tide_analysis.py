"""TIDE error analysis for detection results."""

from pathlib import Path

from tidecv import TIDE, datasets


def run_tide_analysis(
    gt_path: str | Path,
    pred_path: str | Path,
    output_dir: str | Path,
    method_name: str = "detector",
) -> dict:
    """Run TIDE error analysis on detection results.

    TIDE decomposes errors into:
    - Classification error
    - Localization error
    - Both cls + loc error
    - Duplicate detections
    - Background (false positive)
    - Missed (false negative)

    Args:
        gt_path: Path to COCO ground truth JSON.
        pred_path: Path to COCO predictions JSON.
        output_dir: Directory for TIDE output plots.
        method_name: Name of the detection method.

    Returns:
        Dict of error components.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tide = TIDE()
    gt = datasets.COCO(str(gt_path))
    pred = datasets.COCOResult(str(pred_path))

    tide.evaluate(gt, pred, pos_threshold=0.5, mode=TIDE.BOX)
    tide.summarize()

    # Save plots
    tide.plot(str(output_dir / f"{method_name}_tide"))

    errors = {
        "method": method_name,
        "classification": tide.get_main_errors().get("Cls", 0),
        "localization": tide.get_main_errors().get("Loc", 0),
        "cls_and_loc": tide.get_main_errors().get("Both", 0),
        "duplicate": tide.get_main_errors().get("Dupe", 0),
        "background": tide.get_main_errors().get("Bkg", 0),
        "missed": tide.get_main_errors().get("Miss", 0),
    }

    return errors
