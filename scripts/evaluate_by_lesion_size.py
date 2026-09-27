"""Lesion-size-stratified evaluation of predicted tumour masks.

Ground truth: the binary tumour masks of a unified_v2 root; a lesion is one
26-connected component and its size is the equivalent spherical diameter
(ESD) of its volume on the 1 mm grid, binned into 0-5, 5-10, 10-15 and
> 15 mm exactly as in lesion_size_statistics.csv.

Predictions: one NIfTI per case, <predictions>/<case_id>.nii.gz, on the
case's own grid (same shape as tumor_mask.nii.gz).  By default any non-zero
voxel is tumour; for a multi-class label map give --tumour-label (e.g. 2 when
1 is liver and 2 is tumour).

Metrics, overall and per dataset, for every size bin:
  lesions            number of ground-truth lesions in the bin
  detected           lesions whose voxels are covered by the prediction by at
                     least --hit-fraction (default 0.10) of their volume
  sensitivity        detected / lesions (= lesion-level recall), 95 % Wilson CI
  lesion_precision   detected / (detected + false-positive components whose
                     size falls in the bin)
  voxel_recall       covered lesion voxels / lesion voxels of the bin
  lesion_dice_mean   Dice between each lesion and the union of the predicted
                     components that touch it (0 for a missed lesion)
And per scope (overall, per dataset): lesion sensitivity / precision / F1 over
all sizes, voxel-level recall / precision / Dice pooled over the cases, mean
case tumour Dice, false-positive components per case.  Cases with no lesion
contribute only false positives.

Usage:
    python evaluate_by_lesion_size.py --root <unified root> --predictions <dir> \
        --split <split json> --subset test --out <folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

SIZE_BINS = ("0_5mm", "5_10mm", "10_15mm", "gt_15mm")
BIN_LABELS = {"0_5mm": "0-5 mm", "5_10mm": "5-10 mm", "10_15mm": "10-15 mm", "gt_15mm": "> 15 mm"}
STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def esd_mm(volume_mm3: float) -> float:
    return (6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0)


def size_bin(diameter_mm: float) -> str:
    if diameter_mm <= 5.0:
        return "0_5mm"
    if diameter_mm <= 10.0:
        return "5_10mm"
    if diameter_mm <= 15.0:
        return "10_15mm"
    return "gt_15mm"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_mask(path: Path, label: int | None = None) -> tuple[np.ndarray, float]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    array = (data == label) if label is not None else (data > 0)
    voxel_mm3 = float(np.prod(image.header.get_zooms()[:3]))
    return array, voxel_mm3


def evaluate_case(gt: np.ndarray, pred: np.ndarray, voxel_mm3: float, hit_fraction: float, min_pred_voxels: int = 0) -> dict:
    gt_labels, n_gt = ndimage.label(gt, structure=STRUCTURE)
    pred_labels, n_pred = ndimage.label(pred, structure=STRUCTURE)
    if min_pred_voxels > 1 and n_pred:
        # Optional post-processing: drop predicted components below a size,
        # applied identically to every case before matching.
        sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
        keep = sizes >= min_pred_voxels
        keep[0] = False
        pred = keep[pred_labels]
        pred_labels, n_pred = ndimage.label(pred, structure=STRUCTURE)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    lesions = []
    touched_pred: set[int] = set()
    for g in range(1, n_gt + 1):
        region = gt_labels == g
        overlapping = np.unique(pred_labels[region])
        overlapping = [int(p) for p in overlapping if p != 0]
        covered = int(np.count_nonzero(pred[region]))
        union_pred = np.isin(pred_labels, overlapping) if overlapping else np.zeros_like(region)
        dice = 2.0 * covered / (gt_sizes[g] + int(union_pred.sum())) if overlapping else 0.0
        diameter = esd_mm(gt_sizes[g] * voxel_mm3)
        lesions.append(
            {
                "voxels": int(gt_sizes[g]),
                "esd_mm": float(diameter),
                "size_bin": size_bin(diameter),
                "covered_fraction": float(covered / gt_sizes[g]),
                "detected": bool(covered >= max(1, hit_fraction * gt_sizes[g])),
                "lesion_dice": float(dice),
            }
        )
        touched_pred.update(overlapping)
    false_positives = []
    for p in range(1, n_pred + 1):
        if p not in touched_pred:
            diameter = esd_mm(pred_sizes[p] * voxel_mm3)
            false_positives.append({"voxels": int(pred_sizes[p]), "esd_mm": float(diameter), "size_bin": size_bin(diameter)})
    inter = int(np.logical_and(gt, pred).sum())
    denom = int(gt.sum()) + int(pred.sum())
    case_dice = 2.0 * inter / denom if denom else None
    return {"lesions": lesions, "false_positives": false_positives, "case_dice": case_dice,
            "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum()), "tp_voxels": inter}


def aggregate(case_results: list[dict], hit_fraction: float) -> dict:
    out: dict = {"cases": len(case_results), "bins": {}}
    dices = [c["case_dice"] for c in case_results if c["case_dice"] is not None]
    out["case_tumour_dice_mean"] = float(np.mean(dices)) if dices else None
    out["false_positives_per_case"] = float(np.mean([len(c["false_positives"]) for c in case_results])) if case_results else None
    for b in SIZE_BINS:
        lesions = [l for c in case_results for l in c["lesions"] if l["size_bin"] == b]
        fps = [f for c in case_results for f in c["false_positives"] if f["size_bin"] == b]
        detected = sum(l["detected"] for l in lesions)
        lo, hi = wilson(detected, len(lesions))
        gt_vox = sum(l["voxels"] for l in lesions)
        covered_vox = sum(l["covered_fraction"] * l["voxels"] for l in lesions)
        out["bins"][b] = {
            "label": BIN_LABELS[b],
            "lesions": len(lesions),
            "detected": int(detected),
            "sensitivity": float(detected / len(lesions)) if lesions else None,
            "sensitivity_ci95": [lo, hi],
            "lesion_precision": float(detected / (detected + len(fps))) if (detected + len(fps)) else None,
            "voxel_recall": float(covered_vox / gt_vox) if gt_vox else None,
            "lesion_dice_mean": float(np.mean([l["lesion_dice"] for l in lesions])) if lesions else None,
            "false_positive_components": len(fps),
        }
    all_lesions = [l for c in case_results for l in c["lesions"]]
    all_detected = sum(l["detected"] for l in all_lesions)
    all_fps = sum(len(c["false_positives"]) for c in case_results)
    tp = sum(c["tp_voxels"] for c in case_results)
    gt_total = sum(c["gt_voxels"] for c in case_results)
    pred_total = sum(c["pred_voxels"] for c in case_results)
    sens = all_detected / len(all_lesions) if all_lesions else None
    prec = all_detected / (all_detected + all_fps) if (all_detected + all_fps) else None
    out["lesion_level"] = {
        "lesions": len(all_lesions),
        "detected": int(all_detected),
        "sensitivity_recall": sens,
        "precision": prec,
        "f1": (2 * sens * prec / (sens + prec)) if (sens and prec) else None,
        "false_positive_components": int(all_fps),
    }
    out["voxel_level"] = {
        "recall": tp / gt_total if gt_total else None,
        "precision": tp / pred_total if pred_total else None,
        "dice_pooled": 2 * tp / (gt_total + pred_total) if (gt_total + pred_total) else None,
    }
    out["hit_rule"] = f">= {hit_fraction:.2f} of the lesion's voxels covered by the prediction (at least one voxel)"
    out["lesions_total"] = sum(len(c["lesions"]) for c in case_results)
    out["detected_total"] = sum(sum(l["detected"] for l in c["lesions"]) for c in case_results)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Folder with <case_id>.nii.gz binary predictions.")
    parser.add_argument("--split", type=Path, default=None, help="Split JSON from make_splits.py.")
    parser.add_argument("--subset", default="test", help="Subset of the split to evaluate (default test).")
    parser.add_argument("--cases", nargs="*", default=None, help="Explicit case ids instead of a split.")
    parser.add_argument("--hit-fraction", type=float, default=0.10)
    parser.add_argument("--tumour-label", type=int, default=None,
                        help="Label value that means tumour in the prediction files (default: any non-zero voxel).")
    parser.add_argument("--min-pred-voxels", type=int, default=0,
                        help="Drop predicted components smaller than this many voxels (mm3 on the 1 mm grid) before matching; 0 keeps all.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.cases:
        case_ids = list(args.cases)
    elif args.split:
        case_ids = json.loads(args.split.read_text(encoding="utf-8"))["subsets"][args.subset]
    else:
        case_ids = sorted(p.stem.replace(".nii", "") for p in args.predictions.glob("*.nii.gz"))
    manifest = {r["case_id"]: r for r in csv.DictReader((args.root / "dataset_manifest.csv").open(encoding="utf-8-sig", newline=""))}

    per_case = []
    missing = []
    for case_id in case_ids:
        pred_path = args.predictions / f"{case_id}.nii.gz"
        if not pred_path.is_file():
            missing.append(case_id)
            continue
        gt, voxel_mm3 = load_mask(args.root / "cases" / case_id / "tumor_mask.nii.gz")
        pred, _ = load_mask(pred_path, args.tumour_label)
        if pred.shape != gt.shape:
            raise SystemExit(f"{case_id}: prediction shape {pred.shape} != ground truth {gt.shape}")
        result = evaluate_case(gt, pred, voxel_mm3, args.hit_fraction, args.min_pred_voxels)
        result["case_id"] = case_id
        result["dataset"] = manifest[case_id]["dataset"]
        per_case.append(result)
    if missing:
        print(f"WARNING: {len(missing)} cases without a prediction were skipped (first: {missing[:5]})")

    report = {
        "subset": args.subset if args.split else "explicit",
        "cases_evaluated": len(per_case),
        "cases_missing_prediction": missing,
        "hit_fraction": args.hit_fraction,
        "min_pred_voxels": args.min_pred_voxels,
        "tumour_label": args.tumour_label,
        "overall": aggregate(per_case, args.hit_fraction),
        "per_dataset": {d: aggregate([c for c in per_case if c["dataset"] == d], args.hit_fraction) for d in sorted({c["dataset"] for c in per_case})},
    }
    (args.out / "size_binned_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for scope, agg in [("overall", report["overall"])] + list(report["per_dataset"].items()):
        for b in SIZE_BINS:
            e = agg["bins"][b]
            rows.append({"scope": scope, "size_bin": e["label"], "lesions": e["lesions"], "detected": e["detected"],
                         "sensitivity": e["sensitivity"], "ci95_low": e["sensitivity_ci95"][0], "ci95_high": e["sensitivity_ci95"][1],
                         "lesion_precision": e["lesion_precision"], "voxel_recall": e["voxel_recall"],
                         "lesion_dice_mean": e["lesion_dice_mean"], "false_positive_components": e["false_positive_components"],
                         "cases": agg["cases"], "case_tumour_dice_mean": agg["case_tumour_dice_mean"], "false_positives_per_case": agg["false_positives_per_case"],
                         "all_sizes_sensitivity_recall": agg["lesion_level"]["sensitivity_recall"], "all_sizes_precision": agg["lesion_level"]["precision"],
                         "all_sizes_f1": agg["lesion_level"]["f1"], "voxel_recall_pooled": agg["voxel_level"]["recall"],
                         "voxel_precision_pooled": agg["voxel_level"]["precision"], "voxel_dice_pooled": agg["voxel_level"]["dice_pooled"]})
    with (args.out / "size_binned_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "per_lesion.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "dataset", "voxels", "esd_mm", "size_bin", "covered_fraction", "detected", "lesion_dice"])
        writer.writeheader()
        for c in per_case:
            for l in c["lesions"]:
                writer.writerow({"case_id": c["case_id"], "dataset": c["dataset"], **{k: l[k] for k in ("voxels", "esd_mm", "size_bin", "covered_fraction", "detected", "lesion_dice")}})

    print(f"{report['cases_evaluated']} cases; hit rule: {report['overall']['hit_rule']}")
    for scope, agg in [("overall", report["overall"])] + list(report["per_dataset"].items()):
        ll, vl = agg["lesion_level"], agg["voxel_level"]
        fmt = lambda v, d=3: "n/a" if v is None else round(v, d)  # noqa: E731
        print(f"== {scope}: cases {agg['cases']}, case tumour Dice {fmt(agg['case_tumour_dice_mean'])}, FP/case {fmt(agg['false_positives_per_case'], 2)}; "
              f"lesion-level recall {fmt(ll['sensitivity_recall'])} precision {fmt(ll['precision'])} F1 {fmt(ll['f1'])}; "
              f"voxel-level recall {fmt(vl['recall'])} precision {fmt(vl['precision'])} Dice {fmt(vl['dice_pooled'])}")
        for b in SIZE_BINS:
            e = agg["bins"][b]
            sens = "n/a" if e["sensitivity"] is None else f"{e['sensitivity']:.3f} [{e['sensitivity_ci95'][0]:.2f}, {e['sensitivity_ci95'][1]:.2f}]"
            print(f"   {e['label']:9s} lesions {e['lesions']:4d} detected {e['detected']:4d} sensitivity {sens:22s} precision {fmt(e['lesion_precision'])} voxel recall {fmt(e['voxel_recall'])} lesion Dice {fmt(e['lesion_dice_mean'])} FP comps {e['false_positive_components']}")


if __name__ == "__main__":
    main()
