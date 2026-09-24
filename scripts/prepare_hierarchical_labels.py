"""Add hierarchical diagnosis labels and derive lesion-size statistics.

The existing binary tumor masks are read but never modified. Lesion instances are
defined as 26-connected foreground components. Physical volume is converted to an
equivalent spherical diameter (ESD) for the requested non-overlapping size bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


COARSE_IDS = {
    "control": 0,
    "HH": 1,
    "HCC": 2,
    "ICC": 3,
    "cHCC-CCA": 4,
    "metastasis": 5,
}

FINE_IDS = {
    "CN": 0,
    "HH": 1,
    "HCC": 2,
    "ICC": 3,
    "cHCC-CCA": 4,
    "CRLM": 5,
    "BCLM": 6,
}

SIZE_BINS = ("0-5", "5-10", "10-15", ">15")


def label_for_diagnosis(diagnosis: str) -> dict[str, Any]:
    normalized = "cHCC-CCA" if diagnosis == "CHCC" else diagnosis
    if normalized == "CN":
        presence, presence_id = "absent", 0
        behavior, behavior_id = "none", -1
        origin, origin_id = "none", -1
        coarse = "control"
    elif normalized == "HH":
        presence, presence_id = "present", 1
        behavior, behavior_id = "benign", 0
        origin, origin_id = "primary_hepatic", 0
        coarse = "HH"
    elif normalized in {"HCC", "ICC", "cHCC-CCA"}:
        presence, presence_id = "present", 1
        behavior, behavior_id = "malignant", 1
        origin, origin_id = "primary_hepatic", 0
        coarse = normalized
    elif normalized in {"CRLM", "BCLM"}:
        presence, presence_id = "present", 1
        behavior, behavior_id = "malignant", 1
        origin, origin_id = "extrahepatic_metastatic", 1
        coarse = "metastasis"
    else:
        raise ValueError(f"Unsupported diagnosis: {diagnosis}")

    return {
        "lesion_presence": presence,
        "lesion_presence_id": presence_id,
        "behavior": behavior,
        "behavior_id": behavior_id,
        "origin": origin,
        "origin_id": origin_id,
        "diagnosis_coarse": coarse,
        "diagnosis_coarse_id": COARSE_IDS[coarse],
        "diagnosis_fine": normalized,
        "diagnosis_fine_id": FINE_IDS[normalized],
    }


def equivalent_spherical_diameter_mm(volume_mm3: float) -> float:
    if volume_mm3 <= 0:
        raise ValueError("volume_mm3 must be positive")
    return (6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0)


def size_bin(diameter_mm: float) -> str:
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive")
    if diameter_mm <= 5.0:
        return "0-5"
    if diameter_mm <= 10.0:
        return "5-10"
    if diameter_mm <= 15.0:
        return "10-15"
    return ">15"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def component_measurements(mask_path: Path) -> list[dict[str, Any]]:
    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    image = nib.load(str(mask_path))
    binary = np.asanyarray(image.dataobj) > 0
    connected, component_count = ndimage.label(
        binary, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    voxel_volume_mm3 = math.prod(float(v) for v in image.header.get_zooms()[:3])
    voxel_counts = np.bincount(connected.ravel(), minlength=component_count + 1)
    rows = []
    for component_id in range(1, component_count + 1):
        voxel_count = int(voxel_counts[component_id])
        volume_mm3 = voxel_count * voxel_volume_mm3
        diameter_mm = equivalent_spherical_diameter_mm(volume_mm3)
        rows.append(
            {
                "lesion_component_id": int(component_id),
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
                "volume_ml": volume_mm3 / 1000.0,
                "equivalent_spherical_diameter_mm": diameter_mm,
                "size_bin": size_bin(diameter_mm),
            }
        )
    rows.sort(key=lambda row: row["lesion_component_id"])
    return rows


def label_schema() -> dict[str, Any]:
    return {
        "schema_version": "hierarchical_liver_lesion_labels_v1_2026-09-13",
        "segmentation_target": {
            "file": "tumor_mask.nii.gz",
            "values": {"0": "background", "1": "liver_lesion"},
            "note": "Semantic binary mask; disease subtype and instance identity are not encoded in voxel values.",
        },
        "hierarchies": {
            "lesion_presence": {"absent": 0, "present": 1},
            "behavior": {"not_applicable": -1, "benign": 0, "malignant": 1},
            "origin": {
                "not_applicable": -1,
                "primary_hepatic": 0,
                "extrahepatic_metastatic": 1,
            },
            "diagnosis_coarse": COARSE_IDS,
            "diagnosis_fine": FINE_IDS,
        },
        "diagnosis_mapping": {
            "CN": "control without a labeled liver lesion",
            "HH": "hepatic hemangioma; benign primary hepatic lesion",
            "HCC": "hepatocellular carcinoma; primary hepatic malignancy",
            "ICC": "intrahepatic cholangiocarcinoma; primary hepatic malignancy",
            "CHCC": "mapped to cHCC-CCA; combined hepatocellular-cholangiocarcinoma",
            "CRLM": "colorectal liver metastasis; extrahepatic metastatic malignancy",
            "BCLM": "breast cancer liver metastasis; extrahepatic metastatic malignancy",
        },
        "masked_loss_policy": {
            "value": -1,
            "meaning": "not applicable; exclude this target from the corresponding classification loss",
        },
        "lesion_size_definition": {
            "instance_rule": "26-connected component in the unified binary tumor mask",
            "diameter": "equivalent spherical diameter computed from physical component volume",
            "bins_mm": {
                "0-5": "0 < ESD <= 5",
                "5-10": "5 < ESD <= 10",
                "10-15": "10 < ESD <= 15",
                ">15": "ESD > 15",
            },
            "filtering": "No component-size filtering was applied.",
            "clinical_caveat": "ESD is not RECIST longest axial diameter.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Pipeline output root (contains cases/).")
    args = parser.parse_args()
    root = args.root
    cases_root = root / "cases"

    lesion_rows: list[dict[str, Any]] = []
    case_updates: dict[str, dict[str, Any]] = {}
    dataset_cases: Counter[str] = Counter()
    dataset_positive_cases: Counter[str] = Counter()
    dataset_lesions: Counter[tuple[str, str]] = Counter()
    dataset_cases_by_bin: Counter[tuple[str, str]] = Counter()

    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        case_path = case_dir / "case.json"
        metadata = json.loads(case_path.read_text(encoding="utf-8"))
        dataset = metadata["dataset"]
        diagnosis = metadata["diagnosis"]
        labels = label_for_diagnosis(diagnosis)
        components = component_measurements(case_dir / "tumor_mask.nii.gz")
        counts = Counter(component["size_bin"] for component in components)
        diameters = [component["equivalent_spherical_diameter_mm"] for component in components]

        dataset_cases[dataset] += 1
        if components:
            dataset_positive_cases[dataset] += 1
        for bin_name in SIZE_BINS:
            dataset_lesions[(dataset, bin_name)] += counts[bin_name]
            if counts[bin_name] > 0:
                dataset_cases_by_bin[(dataset, bin_name)] += 1

        component_summary = {
            "definition": "26-connected components; equivalent spherical diameter from physical volume",
            "component_filter": "none",
            "lesion_count_total": len(components),
            **{f"lesion_count_{name.replace('>', 'gt_').replace('-', '_')}mm": counts[name] for name in SIZE_BINS},
            "minimum_esd_mm": min(diameters) if diameters else None,
            "median_esd_mm": statistics.median(diameters) if diameters else None,
            "maximum_esd_mm": max(diameters) if diameters else None,
        }
        metadata["classification_labels"] = labels
        metadata["lesion_component_statistics"] = component_summary
        atomic_write_text(case_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

        case_updates[metadata["case_id"]] = {**labels, **component_summary}
        for component in components:
            lesion_rows.append(
                {
                    "dataset": dataset,
                    "patient_id": metadata["patient_id"],
                    "case_id": metadata["case_id"],
                    "diagnosis_raw": diagnosis,
                    "diagnosis_coarse": labels["diagnosis_coarse"],
                    "diagnosis_fine": labels["diagnosis_fine"],
                    **component,
                }
            )

    lesion_fields = [
        "dataset",
        "patient_id",
        "case_id",
        "diagnosis_raw",
        "diagnosis_coarse",
        "diagnosis_fine",
        "lesion_component_id",
        "voxel_count",
        "volume_mm3",
        "volume_ml",
        "equivalent_spherical_diameter_mm",
        "size_bin",
    ]
    atomic_write_csv(root / "lesion_size_statistics.csv", lesion_rows, lesion_fields)

    summary_rows: list[dict[str, Any]] = []
    for dataset in ["MCT-LTDiag", "PLC-CECT", "WAW-TACE"]:
        total_lesions = sum(dataset_lesions[(dataset, name)] for name in SIZE_BINS)
        row: dict[str, Any] = {
            "dataset": dataset,
            "cases_total": dataset_cases[dataset],
            "cases_with_lesion": dataset_positive_cases[dataset],
            "lesions_total": total_lesions,
        }
        for name in SIZE_BINS:
            key = name.replace(">", "gt_").replace("-", "_")
            count = dataset_lesions[(dataset, name)]
            row[f"lesions_{key}mm"] = count
            row[f"lesions_{key}mm_percent"] = 100.0 * count / total_lesions if total_lesions else 0.0
            row[f"cases_with_{key}mm"] = dataset_cases_by_bin[(dataset, name)]
        summary_rows.append(row)

    summary_fields = list(summary_rows[0].keys())
    atomic_write_csv(root / "lesion_size_summary.csv", summary_rows, summary_fields)
    atomic_write_text(
        root / "label_schema.json",
        json.dumps(label_schema(), ensure_ascii=False, indent=2) + "\n",
    )

    manifest_path = root / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        manifest_rows = list(reader)
        manifest_fields = list(reader.fieldnames or [])

    update_fields = [
        "lesion_presence",
        "lesion_presence_id",
        "behavior",
        "behavior_id",
        "origin",
        "origin_id",
        "diagnosis_coarse",
        "diagnosis_coarse_id",
        "diagnosis_fine",
        "diagnosis_fine_id",
        "lesion_count_total",
        "lesion_count_0_5mm",
        "lesion_count_5_10mm",
        "lesion_count_10_15mm",
        "lesion_count_gt_15mm",
        "minimum_esd_mm",
        "median_esd_mm",
        "maximum_esd_mm",
    ]
    for field in update_fields:
        if field not in manifest_fields:
            manifest_fields.append(field)
    for row in manifest_rows:
        update = case_updates[row["case_id"]]
        for field in update_fields:
            row[field] = update.get(field)
    atomic_write_csv(manifest_path, manifest_rows, manifest_fields)

    print(f"Updated {len(case_dirs)} cases")
    print(f"Measured {len(lesion_rows)} connected components")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
