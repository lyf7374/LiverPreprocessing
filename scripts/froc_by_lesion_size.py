"""FROC (sensitivity versus false positives per case) of tumour probability maps, per lesion size.

Input: <probabilities>/<case_id>_prob.nii.gz with the tumour probability
scaled to 0-255 (uint8), as written by predict.py --save-prob, on the case's
own grid.  Ground truth and matching rule as in evaluate_by_lesion_size.py:
lesions are 26-connected components of the tumour mask, binned by equivalent
spherical diameter (0-5, 5-10, 10-15, > 15 mm; 5-15 mm is reported as the
union of the two middle bins); a lesion is detected at threshold t when the
binarised map (probability >= t) covers at least --hit-fraction of its
voxels; a predicted component touching no lesion is a false positive.

Outputs in --out:
  froc.csv            one row per threshold x scope x size bin: lesions,
                      detected, sensitivity, false-positive components of that
                      size, false positives per case (all sizes) at that threshold
  froc_summary.json   per scope and bin: sensitivity interpolated at 0.125,
                      0.25, 0.5, 1, 2, 4, 8 false positives per case (linear
                      interpolation on the operating points; None outside the
                      measured range) and their mean (CPM-style score)

Usage:
    python froc_by_lesion_size.py --root <unified root> --probabilities <dir> \
        --split <split json> --subset test --out <folder> [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_by_lesion_size import BIN_LABELS, SIZE_BINS, evaluate_case, load_mask, wilson  # noqa: E402

THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
FP_TARGETS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
REPORT_BINS = list(SIZE_BINS) + ["5_15mm", "all"]
REPORT_LABELS = {**BIN_LABELS, "5_15mm": "5-15 mm", "all": "all sizes"}


def case_curve(job: tuple) -> dict:
    case_id, root, prob_dir, hit_fraction = job
    gt, voxel_mm3 = load_mask(Path(root) / "cases" / case_id / "tumor_mask.nii.gz")
    prob = np.asanyarray(nib.load(str(Path(prob_dir) / f"{case_id}_prob.nii.gz")).dataobj).astype(np.float32) / 255.0
    out = {"case_id": case_id, "thresholds": {}}
    for t in THRESHOLDS:
        result = evaluate_case(gt, prob >= t, voxel_mm3, hit_fraction)
        out["thresholds"][str(t)] = {
            "lesions": [(l["size_bin"], bool(l["detected"])) for l in result["lesions"]],
            "false_positives": [f["size_bin"] for f in result["false_positives"]],
        }
    return out


def in_bin(size_bin: str, report_bin: str) -> bool:
    if report_bin == "all":
        return True
    if report_bin == "5_15mm":
        return size_bin in ("5_10mm", "10_15mm")
    return size_bin == report_bin


def aggregate(cases: list[dict], datasets: dict[str, str]) -> tuple[list[dict], dict]:
    scopes = ["overall"] + sorted(set(datasets.values()))
    rows, summary = [], {}
    for scope in scopes:
        selected = [c for c in cases if scope == "overall" or datasets[c["case_id"]] == scope]
        summary[scope] = {"cases": len(selected), "bins": {}}
        for report_bin in REPORT_BINS:
            points = []
            for t in THRESHOLDS:
                lesions = detected = fps_bin = fps_all = 0
                for c in selected:
                    r = c["thresholds"][str(t)]
                    for size_bin, det in r["lesions"]:
                        if in_bin(size_bin, report_bin):
                            lesions += 1
                            detected += int(det)
                    fps_bin += sum(in_bin(b, report_bin) for b in r["false_positives"])
                    fps_all += len(r["false_positives"])
                sens = detected / lesions if lesions else None
                fp_per_case = fps_all / len(selected) if selected else None
                lo, hi = wilson(detected, lesions)
                rows.append({"scope": scope, "size_bin": REPORT_LABELS[report_bin], "threshold": t, "lesions": lesions, "detected": detected,
                             "sensitivity": sens, "ci95_low": lo, "ci95_high": hi, "false_positive_components_in_bin": fps_bin,
                             "false_positives_per_case_all_sizes": fp_per_case})
                if sens is not None:
                    points.append((fp_per_case, sens))
            # sensitivity at fixed false-positive rates (interpolate on points sorted by FP/case)
            points.sort()
            at = {}
            for target in FP_TARGETS:
                value = None
                for (x0, y0), (x1, y1) in zip(points, points[1:]):
                    if x0 <= target <= x1:
                        value = y0 if x1 == x0 else y0 + (y1 - y0) * (target - x0) / (x1 - x0)
                        break
                if value is None and points and abs(points[0][0] - target) < 1e-9:
                    value = points[0][1]
                at[str(target)] = value
            measured = [v for v in at.values() if v is not None]
            summary[scope]["bins"][REPORT_LABELS[report_bin]] = {
                "lesions": lesions,
                "sensitivity_at_fp_per_case": at,
                "mean_sensitivity_over_measured_fp_rates": float(np.mean(measured)) if measured else None,
                "fp_per_case_range": [points[0][0], points[-1][0]] if points else None,
            }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--hit-fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    case_ids = json.loads(args.split.read_text(encoding="utf-8"))["subsets"][args.subset]
    missing = [c for c in case_ids if not (args.probabilities / f"{c}_prob.nii.gz").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} probability maps missing (first: {missing[:3]})")
    datasets = {r["case_id"]: r["dataset"] for r in csv.DictReader((args.root / "dataset_manifest.csv").open(encoding="utf-8-sig", newline=""))}
    jobs = [(c, str(args.root), str(args.probabilities), args.hit_fraction) for c in case_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = []
        for index, result in enumerate(pool.map(case_curve, jobs, chunksize=2), start=1):
            cases.append(result)
            if index % 50 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} cases", flush=True)
    rows, summary = aggregate(cases, datasets)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "froc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary["hit_rule"] = f">= {args.hit_fraction:.2f} of the lesion's voxels covered"
    summary["thresholds"] = THRESHOLDS
    (args.out / "froc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for report_bin in ("5-15 mm", "0-5 mm", "> 15 mm", "all sizes"):
        b = summary["overall"]["bins"][report_bin]
        at = {k: (round(v, 3) if v is not None else None) for k, v in b["sensitivity_at_fp_per_case"].items()}
        print(f"overall {report_bin:>9}: lesions {b['lesions']}, sensitivity at FP/case {at}")


if __name__ == "__main__":
    main()
