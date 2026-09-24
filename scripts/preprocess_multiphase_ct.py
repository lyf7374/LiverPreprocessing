"""Build one four-phase liver CT representation from three public datasets.

The source archives are never modified.  Each completed case contains four
LPS-oriented, liver-cropped, 1 x 1 x 1 mm channels in the order NC/AP/PVP/DP
stored as int16 Hounsfield units, a binary liver mask, a binary tumour mask,
the registration transforms of the non-reference phases, and a JSON sidecar.

Intensity convention (unified_v2):
* MCT-LTDiag and WAW-TACE: released HU, stored unchanged (int16, rounded).
* PLC-CECT: the release is an 8-bit window of [-200, 200] HU
  (window level 0 / width 400, confirmed by the data paper and the authors'
  ``window_adjust.py``).  It is mapped back to HU with
  ``HU = v / 255 * 400 - 200``; values above 200 HU are saturated in the source
  and cannot be recovered.  Voxels outside the acquired field of view receive
  -1000 HU.  No clipping or normalisation is applied here; it belongs to the
  data loader.

Spatial convention (unified_v2):
* Every phase is resampled once, directly from its native grid through its
  registration transform, onto a 1 mm isotropic grid that covers the reference
  phase's liver-and-tumour bounding box plus a physical margin.
* MCT-LTDiag uses the released PVP-reference registered grid.
* PLC-CECT uses C2/PVP as the reference phase.  Its released headers carry no
  physical geometry; the in-plane spacing is set to 400 mm / 512 = 0.78125 mm
  and the per-phase slice spacing is taken from ``plc_geometry.csv`` produced
  by ``estimate_plc_geometry.py`` (vertebra-calibrated, +/-10 %).
* WAW-TACE uses PVP as the reference phase.  A tumour annotation drawn in a
  different phase is propagated with that phase's registration transform.
* PLC-CECT and WAW-TACE phases are registered to PVP with a liver-mask driven
  affine stage followed by an intensity-driven (mutual information inside the
  liver) B-spline stage.  The best liver-overlap candidate is kept.  A phase
  whose final liver Dice is below the gate is still written but flagged as not
  usable, so that a model can drop or mask it instead of training on a
  misaligned channel.

Every case records the native geometry of each phase, the origin of its
spacing and intensity values, per-phase registration quality and a set of
geometric sanity flags in ``case.json`` and in ``dataset_manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import traceback
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPENDENCIES = PROJECT_ROOT / "tmp" / "python_deps"
if LOCAL_DEPENDENCIES.exists():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))

try:
    import numpy as np
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - startup diagnostic
    raise SystemExit(
        "SimpleITK and NumPy are required. Run with the bundled Python and "
        f"make {LOCAL_DEPENDENCIES} available. Original error: {exc}"
    ) from exc


PHASES = ("NC", "AP", "PVP", "DP")
PLC_PHASES = {"P": "NC", "C1": "AP", "C2": "PVP", "C3": "DP"}
MCT_FILES = {"NC": "nc.nii.gz", "AP": "art.nii.gz", "PVP": "pvp.nii.gz", "DP": "delay.nii.gz"}
WAW_PHASES = {"0": "NC", "1": "AP", "2": "PVP", "3": "DP"}
WAW_LIVER_LABEL = 5
OUTPUT_FILES = {
    "NC": "image_nc.nii.gz",
    "AP": "image_ap.nii.gz",
    "PVP": "image_pvp.nii.gz",
    "DP": "image_dp.nii.gz",
}
PIPELINE_VERSION = "unified_v2_hu_int16_1mm_iso_gated_registration_2026-09-24"

PLC_IN_PLANE_SPACING_MM = 400.0 / 512.0  # 0.78125 mm: 400 mm display FOV, 512 matrix
PLC_WINDOW_HU = (-200.0, 200.0)
TARGET_SPACING_MM = (1.0, 1.0, 1.0)
OUT_OF_FIELD_HU = -1000.0
REGISTRATION_CLIP_HU = (-200.0, 300.0)
REGISTRATION_DICE_GATE = 0.80
BSPLINE_DICE_TOLERANCE = 0.02
INT16_RANGE = (-32768, 32767)

SANITY_FOV_MM = (250.0, 500.0)
SANITY_SLICE_SPACING_MM = (0.4, 7.0)
SANITY_LIVER_VOLUME_ML = (500.0, 3500.0)
SANITY_PHASE_Z_EXTENT_RATIO = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Folder holding the MCT-LTDiag, PLC-CECT and WAW-TACE release folders.",
    )
    parser.add_argument("--mct-root", type=Path, default=None, help="MCT-LTDiag release folder (default <raw-root>/MCT-LTDiag).")
    parser.add_argument("--plc-root", type=Path, default=None, help="PLC-CECT release folder (default <raw-root>/PLC-CECT).")
    parser.add_argument("--waw-root", type=Path, default=None, help="WAW-TACE release folder (default <raw-root>/WAW-TACE).")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root; cases/ and manifests/ are created inside.")
    parser.add_argument(
        "--waw-selection",
        type=Path,
        default=None,
        help="CSV with a patient_id column; default <output-root>/manifests/waw_four_phase_patients.csv",
    )
    parser.add_argument(
        "--plc-geometry",
        type=Path,
        default=None,
        help="plc_geometry.csv from estimate_plc_geometry.py; default <output-root>/manifests/plc_geometry.csv",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("PLC-CECT", "MCT-LTDiag", "WAW-TACE"),
        default=list(("PLC-CECT", "MCT-LTDiag", "WAW-TACE")),
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Process only the first N cases from each selected dataset.",
    )
    parser.add_argument("--patients", nargs="*", default=None, help="Only these patient ids.")
    parser.add_argument("--crop-margin-mm", type=float, default=20.0)
    parser.add_argument(
        "--replace-datasets",
        nargs="*",
        choices=("PLC-CECT", "MCT-LTDiag", "WAW-TACE"),
        default=[],
        help=(
            "Reprocess and atomically replace completed cases from these datasets "
            "whose case.json does not carry the current pipeline version."
        ),
    )
    parser.add_argument(
        "--registration-spacing-mm",
        type=float,
        default=4.0,
        help="Minimum per-axis spacing used only during registration optimization.",
    )
    parser.add_argument(
        "--bspline-grid-spacing-mm",
        type=float,
        default=40.0,
        help="Approximate B-spline control-point spacing in physical millimetres.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #


def path_source(path: Path) -> dict[str, str]:
    return {"kind": "path", "path": str(path)}


def archive_source(kind: str, archive: Path, member: str) -> dict[str, str]:
    return {"kind": kind, "archive": str(archive), "member": member}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def zip_member_index(archives: Iterable[Path]) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for archive in sorted(archives):
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                basename = Path(info.filename).name
                if basename in index:
                    raise ValueError(f"Duplicate archive member basename: {basename}")
                index[basename] = (archive, info.filename)
    return index


def required_zip_source(
    index: dict[str, tuple[Path, str]], basename: str
) -> dict[str, str]:
    if basename not in index:
        raise FileNotFoundError(f"Archive member not found: {basename}")
    archive, member = index[basename]
    return archive_source("zip", archive, member)


def read_plc_geometry(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    geometry: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        geometry[(row["patient_id"], row["phase"])] = {
            "spacing_mm": [
                float(row["spacing_x_mm"]),
                float(row["spacing_y_mm"]),
                float(row["spacing_z_mm"]),
            ],
            "k_axis_flip": bool(int(row.get("k_axis_flip") or 0)),
            "orientation_source": row.get("orientation_source", ""),
            "orientation_evidence": row.get("orientation_evidence", ""),
            "spacing_z_source": row["spacing_z_source"],
            "spacing_z_confidence": row["spacing_z_confidence"],
            "spacing_z_continuous_mm": float(row["spacing_z_continuous_mm"]),
            "spacing_z_snap": row["spacing_z_snap"],
            "liver_z_extent_mm": float(row["liver_z_extent_mm"]),
            "vertebra_pairs": int(row["vertebra_pairs"] or 0),
            "geometry_version": row["geometry_version"],
            "liver_fallback_file": row.get("totalsegmentator_liver_native_file") or None,
        }
    return geometry


DATASET_FOLDERS = {"MCT-LTDiag": "MCT-LTDiag", "PLC-CECT": "PLC-CECT", "WAW-TACE": "WAW-TACE"}


def resolve_dataset_root(dataset: str, raw_root: Path | None, explicit: Path | None) -> Path:
    """Per-dataset release folder: an explicit path wins, else <raw-root>/<dataset>."""
    if explicit is not None:
        root = explicit
    elif raw_root is not None:
        root = raw_root / DATASET_FOLDERS[dataset]
    else:
        raise SystemExit(f"{dataset}: give --raw-root or the dataset's own --*-root option")
    if not root.is_dir():
        raise SystemExit(f"{dataset}: release folder not found: {root}")
    return root


def build_plc_cases(
    root: Path, geometry_csv: Path | None, patients: set[str] | None = None
) -> list[dict[str, Any]]:
    rows = list(csv.DictReader((root / "patient_data.csv").open(encoding="utf-8-sig", newline="")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if patients is None or row["patient_id"] in patients:
            grouped[row["patient_id"]].append(row)
    index = zip_member_index(root.glob("*.zip"))
    geometry = read_plc_geometry(geometry_csv) if geometry_csv is not None else None
    cases: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        by_phase = {PLC_PHASES[row["phase"]]: row for row in grouped[patient_id]}
        if set(by_phase) != set(PHASES):
            raise ValueError(f"PLC case {patient_id} does not have exactly four phases")
        phase_sources: dict[str, dict[str, str]] = {}
        liver_sources: dict[str, dict[str, str]] = {}
        liver_mask_origin: dict[str, str] = {}
        phase_geometry: dict[str, dict[str, Any]] = {}
        for phase in PHASES:
            row = by_phase[phase]
            phase_sources[phase] = required_zip_source(index, Path(row["ct_path"]).name)
            liver_sources[phase] = required_zip_source(index, Path(row["liver_mask_path"]).name)
            liver_mask_origin[phase] = "released"
            if geometry is not None:
                key = (patient_id, phase)
                if key not in geometry:
                    raise KeyError(f"plc_geometry.csv has no row for {patient_id} {phase}")
                phase_geometry[phase] = geometry[key]
                fallback = geometry[key]["liver_fallback_file"]
                if fallback:
                    liver_sources[phase] = path_source(Path(fallback))
                    liver_mask_origin[phase] = "totalsegmentator_3mm_fallback"
        reference_row = by_phase["PVP"]
        tumour_sources_by_phase: dict[str, list[dict[str, str]]] = {}
        for phase, row in by_phase.items():
            mask_value = row.get("mask_path", "").strip()
            tumour_sources_by_phase[phase] = []
            if mask_value.upper() not in {"", "NA", "N/A", "NONE", "NULL"}:
                tumour_sources_by_phase[phase] = [
                    required_zip_source(index, Path(mask_value).name)
                ]
        cases.append(
            {
                "dataset": "PLC-CECT",
                "patient_id": patient_id,
                "case_id": f"PLC-CECT_{patient_id}",
                "diagnosis": reference_row["cancer_type"],
                "reference_phase": "PVP",
                "phase_sources": phase_sources,
                "liver_sources": liver_sources,
                "liver_mask_origin": liver_mask_origin,
                "tumour_sources": tumour_sources_by_phase["PVP"],
                "tumour_sources_by_phase": tumour_sources_by_phase,
                "intensity_kind": "released_uint8_windowed",
                "registration_policy": "liver_mask_affine_then_mi_bspline_to_pvp_with_dice_gate",
                "phase_geometry": phase_geometry,
            }
        )
    return cases


def tar_member_names(archive: Path) -> dict[str, str]:
    wanted = set(MCT_FILES.values()) | {"mask_pvp.nii.gz", "liver_mask_pvp.nii.gz"}
    found: dict[str, str] = {}
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            basename = Path(member.name).name
            if member.isfile() and basename in wanted:
                found[basename] = member.name
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"{archive.name} is missing {sorted(missing)}")
    return found


def build_mct_cases(root: Path) -> list[dict[str, Any]]:
    metadata_rows = list(
        csv.DictReader(
            (root / "meta_info_patient.tab").open(encoding="utf-8-sig", newline=""),
            delimiter="\t",
        )
    )
    metadata = {row["ID"]: row for row in metadata_rows}
    cases: list[dict[str, Any]] = []
    for archive in sorted(root.glob("*.tar")):
        patient_id = archive.stem
        members = {
            **{basename: f"./NIFTI/{basename}" for basename in MCT_FILES.values()},
            "mask_pvp.nii.gz": "./mask_pvp.nii.gz",
            "liver_mask_pvp.nii.gz": "./liver_mask_pvp.nii.gz",
        }
        row = metadata.get(patient_id, {})
        cases.append(
            {
                "dataset": "MCT-LTDiag",
                "patient_id": patient_id,
                "case_id": f"MCT-LTDiag_{patient_id}",
                "diagnosis": row.get("type", ""),
                "reference_phase": "PVP",
                "phase_sources": {
                    phase: archive_source("tar", archive, members[basename])
                    for phase, basename in MCT_FILES.items()
                },
                "liver_sources": {
                    "PVP": archive_source("tar", archive, members["liver_mask_pvp.nii.gz"])
                },
                "liver_mask_origin": {"PVP": "released"},
                "tumour_sources": [
                    archive_source("tar", archive, members["mask_pvp.nii.gz"])
                ],
                "intensity_kind": "hu",
                "registration_policy": "official_pvp_reference_registration",
                "phase_geometry": {},
            }
        )
    return cases


def build_waw_mask_index(mask_archive: Path) -> dict[str, list[dict[str, str]]]:
    pattern = re.compile(r"/(\d+)_(\d+)_(\d+)_tumor_seg\.nrrd$")
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    with zipfile.ZipFile(mask_archive) as handle:
        for info in handle.infolist():
            match = pattern.search(info.filename)
            if not match:
                continue
            basename = Path(info.filename).name
            if basename in seen:
                continue
            seen.add(basename)
            by_patient[match.group(1)].append(
                {
                    **archive_source("zip", mask_archive, info.filename),
                    "phase_index": match.group(2),
                    "lesion_index": match.group(3),
                }
            )
    return by_patient


def build_waw_cases(root: Path, selection_path: Path) -> list[dict[str, Any]]:
    image_root = root / "huggingface" / "images"
    organ_root = root / "huggingface" / "organ_masks"
    selected = {
        row["patient_id"]
        for row in csv.DictReader(selection_path.open(encoding="utf-8-sig", newline=""))
    }
    scan_rows = list(
        csv.DictReader((root / "ct_hcc_metadata.csv").open(encoding="utf-8-sig", newline=""))
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scan_rows:
        if row["patient_id"] in selected:
            grouped[row["patient_id"]].append(row)
    mask_index = build_waw_mask_index(root / "tumor_masks.zip")
    cases: list[dict[str, Any]] = []
    for patient_id in sorted(selected, key=int):
        by_phase = {WAW_PHASES[row["ct_phase"]]: row for row in grouped[patient_id]}
        if set(by_phase) != set(PHASES):
            raise ValueError(f"WAW case {patient_id} does not have exactly four phases")
        tumour_sources = mask_index.get(patient_id, [])
        annotated_indices = {source["phase_index"] for source in tumour_sources}
        if len(annotated_indices) != 1:
            raise ValueError(
                f"WAW case {patient_id} must have one annotated phase, got {annotated_indices}"
            )
        annotated_phase = WAW_PHASES[next(iter(annotated_indices))]
        cases.append(
            {
                "dataset": "WAW-TACE",
                "patient_id": patient_id,
                "case_id": f"WAW-TACE_{patient_id}",
                "diagnosis": "HCC",
                "reference_phase": "PVP",
                "tumour_annotation_phase": annotated_phase,
                "phase_sources": {
                    phase: path_source(image_root / f"{row['ct_file_name']}.nii.gz")
                    for phase, row in by_phase.items()
                },
                "liver_sources": {
                    phase: path_source(organ_root / f"{row['ct_file_name']}.nii.gz")
                    for phase, row in by_phase.items()
                },
                "liver_mask_origin": {phase: "released_organ_mask_label_5" for phase in PHASES},
                "tumour_sources": tumour_sources,
                "intensity_kind": "hu",
                "registration_policy": "liver_mask_affine_then_mi_bspline_to_pvp_with_dice_gate",
                "phase_geometry": {},
                "released_slice_thickness_mm": {
                    phase: float(row["slice_thickness"]) for phase, row in by_phase.items()
                },
            }
        )
    return cases


# --------------------------------------------------------------------------- #
# Image reading and conventions
# --------------------------------------------------------------------------- #


def materialize(source: dict[str, str], directory: Path, name: str) -> Path:
    if source["kind"] == "path":
        path = Path(source["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty source file: {path}")
        return path
    suffix = ".nii.gz" if source["member"].endswith(".nii.gz") else Path(source["member"]).suffix
    destination = directory / f"{name}{suffix}"
    if source["kind"] == "zip":
        with zipfile.ZipFile(source["archive"]) as handle:
            info = handle.getinfo(source["member"])
            if info.file_size == 0:
                raise ValueError(f"Empty ZIP member: {source['member']}")
            with handle.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    elif source["kind"] == "tar":
        with tarfile.open(source["archive"], "r") as handle:
            member = handle.getmember(source["member"])
            if member.size == 0:
                raise ValueError(f"Empty TAR member: {source['member']}")
            src = handle.extractfile(member)
            if src is None:
                raise ValueError(f"Cannot read TAR member: {source['member']}")
            with src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    else:
        raise ValueError(f"Unsupported source kind: {source['kind']}")
    return destination


def read_scalar_image(path: Path) -> tuple[sitk.Image, dict[str, Any]]:
    image = sitk.ReadImage(str(path))
    original = {
        "dimension": image.GetDimension(),
        "components": image.GetNumberOfComponentsPerPixel(),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "header_spacing_mm": list(image.GetSpacing()),
        "header_origin_mm": list(image.GetOrigin()),
    }
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got {image.GetDimension()}D: {path}")
    if image.GetNumberOfComponentsPerPixel() > 1:
        components = [
            sitk.VectorIndexSelectionCast(image, index)
            for index in range(image.GetNumberOfComponentsPerPixel())
        ]
        first = sitk.GetArrayViewFromImage(components[0])
        if all(np.array_equal(first, sitk.GetArrayViewFromImage(item)) for item in components[1:]):
            image = components[0]
            original["vector_fix"] = "identical_components_collapsed_to_first"
        else:
            # Some PLC volumes were released as RGB renderings of a greyscale CT.
            # Averaging the channels reverses that representation.  The liver crop
            # removes the coloured ruler marks at the image border.
            array = sitk.GetArrayFromImage(image).astype(np.float32).mean(axis=-1)
            scalar = sitk.GetImageFromArray(array)
            scalar.CopyInformation(components[0])
            image = scalar
            original["vector_fix"] = "nonidentical_rgb_components_averaged_to_greyscale"
    else:
        original["vector_fix"] = None
    image = sitk.DICOMOrient(image, "LPS")
    return image, original


def flip_k_axis(image: sitk.Image) -> sitk.Image:
    """Reverse the slice order in place on the same grid.

    The released PLC-CECT headers declare k -> superior, but the slices of most
    volumes are stored superior -> inferior.  Reversing the array while keeping
    the grid makes k -> superior true, as it is for MCT-LTDiag and WAW-TACE.
    """
    array = sitk.GetArrayFromImage(image)[::-1].copy()
    flipped = sitk.GetImageFromArray(array)
    flipped.CopyInformation(image)
    return flipped


def apply_phase_geometry(image: sitk.Image, phase_geometry: dict[str, Any] | None) -> sitk.Image:
    """Overwrite the header spacing of a PLC volume with the estimated geometry
    and reverse its slice order when the release stored it inferior-first."""
    if phase_geometry is None:
        return image
    image.SetSpacing(tuple(float(v) for v in phase_geometry["spacing_mm"]))
    if phase_geometry.get("k_axis_flip"):
        image = flip_k_axis(image)
    return image


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size": list(image.GetSize()),
        "spacing_mm": list(image.GetSpacing()),
        "origin_mm": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def same_grid(left: sitk.Image, right: sitk.Image, tolerance: float = 1e-4) -> bool:
    return left.GetSize() == right.GetSize() and all(
        np.allclose(a, b, atol=tolerance, rtol=0)
        for a, b in (
            (left.GetSpacing(), right.GetSpacing()),
            (left.GetOrigin(), right.GetOrigin()),
            (left.GetDirection(), right.GetDirection()),
        )
    )


def binary_mask(image: sitk.Image, label: int | None = None) -> sitk.Image:
    result = image == label if label is not None else image > 0
    return sitk.Cast(result, sitk.sitkUInt8)


def align_mask_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if same_grid(mask, reference):
        aligned = sitk.Cast(mask > 0, sitk.sitkUInt8)
        aligned.CopyInformation(reference)
        return aligned
    return sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def plc_uint8_to_hu(image: sitk.Image) -> sitk.Image:
    low, high = PLC_WINDOW_HU
    image = sitk.Clamp(sitk.Cast(image, sitk.sitkFloat32), lowerBound=0.0, upperBound=255.0)
    return sitk.Cast(image / 255.0 * (high - low) + low, sitk.sitkFloat32)


def to_hounsfield(image: sitk.Image, intensity_kind: str) -> sitk.Image:
    if intensity_kind == "hu":
        return sitk.Cast(image, sitk.sitkFloat32)
    if intensity_kind == "released_uint8_windowed":
        return plc_uint8_to_hu(image)
    raise ValueError(f"Unknown intensity kind: {intensity_kind}")


def registration_intensity(image: sitk.Image) -> sitk.Image:
    low, high = REGISTRATION_CLIP_HU
    return sitk.Clamp(image, lowerBound=low, upperBound=high)


def mask_has_foreground(mask: sitk.Image) -> bool:
    return int(sitk.GetArrayViewFromImage(mask).sum()) > 0


def mask_volume_ml(mask: sitk.Image) -> float:
    voxels = int(sitk.GetArrayViewFromImage(mask).sum())
    return voxels * math.prod(mask.GetSpacing()) / 1000.0


def mask_z_extent_mm(mask: sitk.Image) -> float:
    array = sitk.GetArrayViewFromImage(mask) > 0
    slices = np.flatnonzero(array.any(axis=(1, 2)))
    if slices.size == 0:
        return 0.0
    return float((slices[-1] - slices[0] + 1) * mask.GetSpacing()[2])


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def liver_centroid_initializer(
    fixed_liver: sitk.Image, moving_liver: sitk.Image
) -> tuple[sitk.Euler3DTransform, dict[str, Any]]:
    transform = sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            sitk.Cast(fixed_liver, sitk.sitkFloat32),
            sitk.Cast(moving_liver, sitk.sitkFloat32),
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
    )
    parameters = transform.GetParameters()
    return transform, {
        "method": "Euler3D liver-centroid translation",
        "translation_mm": list(parameters[3:]),
    }


def geometry_center_register(
    fixed: sitk.Image,
    moving: sitk.Image,
) -> tuple[sitk.Transform, dict[str, Any]]:
    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    parameters = transform.GetParameters()
    return transform, {
        "method": "Euler3D geometry-centre translation",
        "selected_stage": "geometry_centre",
        "translation_mm": list(parameters[3:]),
        "liver_dice_by_stage": {},
        "final_liver_dice": None,
        "note": "Used because a liver mask was empty; no overlap could be measured.",
    }


def resample_registration_pair(
    image: sitk.Image,
    mask: sitk.Image,
    minimum_spacing_mm: float,
) -> tuple[sitk.Image, sitk.Image]:
    spacing = tuple(
        max(float(value), minimum_spacing_mm) for value in image.GetSpacing()
    )
    size = [
        max(1, int(round(n * old / new)))
        for n, old, new in zip(image.GetSize(), image.GetSpacing(), spacing)
    ]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(spacing)
    grid.SetOrigin(image.GetOrigin())
    grid.SetDirection(image.GetDirection())
    image_out = sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32),
        grid,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        REGISTRATION_CLIP_HU[0],
        sitk.sitkFloat32,
    )
    mask_out = sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        grid,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return image_out, mask_out


def registration_roi(
    image: sitk.Image,
    mask: sitk.Image,
    minimum_spacing_mm: float,
    margin_mm: float = 30.0,
) -> tuple[sitk.Image, sitk.Image]:
    start, size = crop_region(mask, margin_mm)
    return resample_registration_pair(
        crop(image, start, size),
        crop(mask, start, size),
        minimum_spacing_mm,
    )


def transformed_mask_dice(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:
    warped = sitk.Resample(
        sitk.Cast(moving_mask > 0, sitk.sitkUInt8),
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    fixed_array = sitk.GetArrayViewFromImage(fixed_mask) > 0
    warped_array = sitk.GetArrayViewFromImage(warped) > 0
    denominator = int(fixed_array.sum()) + int(warped_array.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(fixed_array, warped_array).sum() / denominator)


def smoothed_mask(mask: sitk.Image, sigma_mm: float) -> sitk.Image:
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(mask, sitk.sitkFloat32), sigma_mm)


def mask_register(
    fixed_mask_reg: sitk.Image,
    moving_mask_reg: sitk.Image,
    transform: sitk.Transform,
) -> tuple[sitk.Transform, dict[str, Any]]:
    """Align the two liver masks (mean squares on Gaussian-smoothed masks).

    The masks are available in every phase, so this stage cannot collapse the
    way an intensity metric with a moving-image mask can.  With a rigid
    transform it absorbs the breathing shift and small rotations; with an
    affine transform it additionally absorbs, for PLC-CECT, residual per-phase
    slice-spacing error as a z scale.
    """
    fixed_soft = smoothed_mask(fixed_mask_reg, 4.0)
    moving_soft = smoothed_mask(moving_mask_reg, 4.0)
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMeanSquares()
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=150,
        relaxationFactor=0.6,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([8.0, 4.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(transform, inPlace=True)
    method.Execute(fixed_soft, moving_soft)
    details: dict[str, Any] = {
        "metric_value": float(method.GetMetricValue()),
        "stop_condition": method.GetOptimizerStopConditionDescription(),
    }
    if isinstance(transform, sitk.AffineTransform):
        matrix = np.array(transform.GetMatrix()).reshape(3, 3)
        details["axis_scales"] = [float(np.linalg.norm(matrix[:, axis])) for axis in range(3)]
    return transform, details


def rigid_from_initial(initial: sitk.Euler3DTransform) -> sitk.Euler3DTransform:
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(initial.GetCenter())
    rigid.SetTranslation(initial.GetTranslation())
    return rigid


def affine_from_initial(initial: sitk.Euler3DTransform) -> sitk.AffineTransform:
    affine = sitk.AffineTransform(3)
    affine.SetCenter(initial.GetCenter())
    affine.SetMatrix(initial.GetMatrix())
    affine.SetTranslation(initial.GetTranslation())
    return affine


AFFINE_SCALE_RANGE = (0.85, 1.18)


def liver_guided_affine_bspline_register(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_liver: sitk.Image,
    moving_liver: sitk.Image,
    registration_spacing_mm: float,
    bspline_grid_spacing_mm: float,
) -> tuple[sitk.Transform, dict[str, Any]]:
    if not mask_has_foreground(fixed_liver) or not mask_has_foreground(moving_liver):
        return geometry_center_register(fixed, moving)

    initial, initial_details = liver_centroid_initializer(fixed_liver, moving_liver)
    fixed_reg, fixed_mask_reg = registration_roi(
        registration_intensity(fixed), fixed_liver, registration_spacing_mm
    )
    moving_reg, moving_mask_reg = registration_roi(
        registration_intensity(moving), moving_liver, registration_spacing_mm
    )

    candidates: dict[str, sitk.Transform] = {"centroid": initial}
    rigid_details: dict[str, Any] = {}
    affine_details: dict[str, Any] = {}
    affine_error: str | None = None
    try:
        rigid, rigid_details = mask_register(fixed_mask_reg, moving_mask_reg, rigid_from_initial(initial))
        candidates["rigid"] = rigid
    except RuntimeError as exc:
        rigid_details = {"error": str(exc)}
    try:
        affine, affine_details = mask_register(fixed_mask_reg, moving_mask_reg, affine_from_initial(initial))
        scales = affine_details["axis_scales"]
        if all(AFFINE_SCALE_RANGE[0] <= scale <= AFFINE_SCALE_RANGE[1] for scale in scales):
            candidates["affine"] = affine
        else:
            # A scale this far from 1 means the two liver masks do not cover the
            # same anatomy (truncated scan, breathing, segmentation), not a
            # spacing error; the rigid candidate is kept instead.
            affine_details["rejected"] = f"axis scale outside {AFFINE_SCALE_RANGE}"
    except RuntimeError as exc:
        affine_error = str(exc)

    initial_scores = {
        name: transformed_mask_dice(fixed_liver, moving_liver, transform)
        for name, transform in candidates.items()
    }
    deformable_base_stage = max(initial_scores, key=initial_scores.get)
    deformable_base = candidates[deformable_base_stage]
    deformable_error: str | None = None
    deformable_metric: float | None = None
    deformable_stop: str | None = None
    mesh_size: list[int] | None = None
    try:
        base_warped = sitk.Resample(
            moving_reg,
            fixed_reg,
            deformable_base,
            sitk.sitkLinear,
            REGISTRATION_CLIP_HU[0],
            sitk.sitkFloat32,
        )
        dilation = max(1, int(round(10.0 / registration_spacing_mm)))
        fixed_metric_mask = sitk.BinaryDilate(fixed_mask_reg, [dilation] * 3)
        physical_size = [
            max(1.0, (n - 1) * spacing)
            for n, spacing in zip(fixed_reg.GetSize(), fixed_reg.GetSpacing())
        ]
        mesh_size = [
            max(1, int(round(length / bspline_grid_spacing_mm)))
            for length in physical_size
        ]
        bspline = sitk.BSplineTransformInitializer(fixed_reg, mesh_size, order=3)
        deformable_method = sitk.ImageRegistrationMethod()
        deformable_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        deformable_method.SetMetricSamplingStrategy(deformable_method.RANDOM)
        deformable_method.SetMetricSamplingPercentage(0.20, 43)
        deformable_method.SetMetricFixedMask(fixed_metric_mask)
        deformable_method.SetInterpolator(sitk.sitkLinear)
        deformable_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=40,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=200,
            costFunctionConvergenceFactor=1e7,
        )
        deformable_method.SetShrinkFactorsPerLevel([2, 1])
        deformable_method.SetSmoothingSigmasPerLevel([1.0, 0.0])
        deformable_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        deformable_method.SetInitialTransform(bspline, inPlace=True)
        deformable_method.Execute(fixed_reg, base_warped)
        deformable_metric = float(deformable_method.GetMetricValue())
        deformable_stop = deformable_method.GetOptimizerStopConditionDescription()
        composite = sitk.CompositeTransform(3)
        # CompositeTransform applies the most recently added transform first:
        # fixed point -> B-spline point -> original moving-image point.
        composite.AddTransform(deformable_base)
        composite.AddTransform(bspline)
        candidates[f"{deformable_base_stage}_bspline"] = composite
    except RuntimeError as exc:
        deformable_error = str(exc)

    scores = {
        name: transformed_mask_dice(fixed_liver, moving_liver, transform)
        for name, transform in candidates.items()
    }
    deformable_stage = f"{deformable_base_stage}_bspline"
    if (
        deformable_stage in scores
        and scores[deformable_stage] >= scores[deformable_base_stage] - BSPLINE_DICE_TOLERANCE
    ):
        # The B-spline stage aligns internal structures; a liver-boundary Dice
        # within tolerance of its rigid/affine base is not a reason to reject it.
        selected_stage = deformable_stage
    else:
        selected_stage = max(scores, key=scores.get)
    selected = candidates[selected_stage]
    return selected, {
        "method": "liver-mask rigid/affine plus mutual-information B-spline registration",
        "initial": initial_details,
        "selected_stage": selected_stage,
        "deformable_base_stage": deformable_base_stage,
        "liver_dice_by_stage": scores,
        "final_liver_dice": scores[selected_stage],
        "registration_spacing_mm": registration_spacing_mm,
        "bspline_grid_spacing_mm": bspline_grid_spacing_mm,
        "bspline_mesh_size": mesh_size,
        "rigid": rigid_details,
        "affine": affine_details,
        "affine_error": affine_error,
        "deformable_metric_value": deformable_metric,
        "deformable_stop_condition": deformable_stop,
        "deformable_error": deformable_error,
        "selection_note": (
            "Candidates: liver-centroid translation, liver-mask rigid, liver-mask affine "
            f"(only with axis scales within {AFFINE_SCALE_RANGE}), and B-spline on top of "
            "the best of them. The B-spline result is kept when its liver Dice is within "
            f"{BSPLINE_DICE_TOLERANCE} of its base; otherwise the highest liver Dice wins. "
            "Tumour labels never influence the transform."
        ),
    }


# --------------------------------------------------------------------------- #
# Cropping, output grid, writing
# --------------------------------------------------------------------------- #


def crop_region(mask: sitk.Image, margin_mm: float) -> tuple[list[int], list[int]]:
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not statistics.HasLabel(1):
        raise ValueError("Reference liver/tumour mask is empty; cannot define a liver crop")
    bounding = statistics.GetBoundingBox(1)
    start = list(bounding[:3])
    size = list(bounding[3:])
    full_size = list(mask.GetSize())
    spacing = mask.GetSpacing()
    for axis in range(3):
        padding = int(math.ceil(margin_mm / spacing[axis]))
        lower = max(0, start[axis] - padding)
        upper = min(full_size[axis], start[axis] + size[axis] + padding)
        start[axis] = lower
        size[axis] = upper - lower
    return start, size


def crop(image: sitk.Image, start: list[int], size: list[int]) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=size, index=start)


def isotropic_output_grid(
    reference: sitk.Image,
    start: list[int],
    size: list[int],
    target_spacing: tuple[float, float, float] = TARGET_SPACING_MM,
) -> sitk.Image:
    """1 mm grid covering the physical extent of a crop of the reference grid.

    All inputs are LPS-oriented with an identity direction, so the grid is
    axis-aligned and its origin is the physical position of the crop's first
    voxel centre.  The last voxel centre of the crop is preserved within one
    target voxel.
    """
    if not np.allclose(reference.GetDirection(), np.eye(3).ravel(), atol=1e-6):
        raise ValueError("Reference grid must have an identity direction after LPS orientation")
    origin = reference.TransformIndexToPhysicalPoint([int(v) for v in start])
    extent_mm = [
        max(0.0, (n - 1) * spacing)
        for n, spacing in zip(size, reference.GetSpacing())
    ]
    out_size = [int(math.floor(extent / target + 1e-6)) + 1 for extent, target in zip(extent_mm, target_spacing)]
    grid = sitk.Image(out_size, sitk.sitkFloat32)
    grid.SetSpacing(target_spacing)
    grid.SetOrigin(origin)
    grid.SetDirection(reference.GetDirection())
    return grid


def resample_image_to_grid(image: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Resample(
        sitk.Cast(image, sitk.sitkFloat32),
        grid,
        transform,
        sitk.sitkLinear,
        OUT_OF_FIELD_HU,
        sitk.sitkFloat32,
    )


def resample_mask_to_grid(mask: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        grid,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def to_int16_hu(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image)
    array = np.rint(array).clip(*INT16_RANGE).astype(np.int16)
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(image)
    return result


def write_compressed_image(image: sitk.Image, path: Path) -> None:
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(True)
    writer.SetCompressionLevel(1)
    writer.Execute(image)


def output_is_complete(case_dir: Path) -> bool:
    required = [*OUTPUT_FILES.values(), "liver_mask.nii.gz", "tumor_mask.nii.gz", "case.json"]
    return all((case_dir / name).is_file() and (case_dir / name).stat().st_size > 0 for name in required)


def output_has_current_pipeline(case_dir: Path) -> bool:
    if not output_is_complete(case_dir):
        return False
    try:
        record = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("pipeline_version") == PIPELINE_VERSION


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #


def sanity_checks(
    case: dict[str, Any],
    native: dict[str, sitk.Image],
    liver_masks: dict[str, sitk.Image],
    reference_phase: str,
    liver_volume_ml: float,
    registrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    flags: list[str] = []
    reference = native[reference_phase]
    fov = [n * s for n, s in zip(reference.GetSize()[:2], reference.GetSpacing()[:2])]
    for axis, value in zip("xy", fov):
        if not SANITY_FOV_MM[0] <= value <= SANITY_FOV_MM[1]:
            flags.append(f"fov_{axis}_{value:.0f}mm_outside_{SANITY_FOV_MM[0]:.0f}-{SANITY_FOV_MM[1]:.0f}")
    for phase, image in native.items():
        spacing_z = image.GetSpacing()[2]
        if not SANITY_SLICE_SPACING_MM[0] <= spacing_z <= SANITY_SLICE_SPACING_MM[1]:
            flags.append(f"{phase}_slice_spacing_{spacing_z:.2f}mm_outside_range")
        if all(abs(v) < 1e-9 for v in image.GetOrigin()) and case["dataset"] != "PLC-CECT":
            flags.append(f"{phase}_zero_origin")
    if liver_volume_ml <= 0:
        flags.append("liver_mask_empty")
    elif not SANITY_LIVER_VOLUME_ML[0] <= liver_volume_ml <= SANITY_LIVER_VOLUME_ML[1]:
        flags.append(f"liver_volume_{liver_volume_ml:.0f}ml_outside_range")
    reference_extent = mask_z_extent_mm(liver_masks[reference_phase]) if reference_phase in liver_masks else 0.0
    extents: dict[str, float] = {}
    for phase, mask in liver_masks.items():
        extent = mask_z_extent_mm(mask)
        extents[phase] = extent
        if phase == reference_phase or reference_extent <= 0 or extent <= 0:
            continue
        ratio = abs(extent - reference_extent) / reference_extent
        if ratio > SANITY_PHASE_Z_EXTENT_RATIO:
            flags.append(f"{phase}_liver_z_extent_differs_{ratio:.2f}_from_{reference_phase}")
    for phase, details in registrations.items():
        dice = details.get("final_liver_dice")
        if phase != reference_phase and dice is not None and dice < REGISTRATION_DICE_GATE:
            flags.append(f"{phase}_registration_dice_{dice:.2f}_below_gate")
    return {
        "pass": not flags,
        "flags": flags,
        "reference_fov_mm": fov,
        "liver_volume_ml": liver_volume_ml,
        "liver_z_extent_mm_by_phase": extents,
        "thresholds": {
            "fov_mm": list(SANITY_FOV_MM),
            "slice_spacing_mm": list(SANITY_SLICE_SPACING_MM),
            "liver_volume_ml": list(SANITY_LIVER_VOLUME_ML),
            "phase_liver_z_extent_ratio": SANITY_PHASE_Z_EXTENT_RATIO,
            "registration_dice_gate": REGISTRATION_DICE_GATE,
        },
    }


# --------------------------------------------------------------------------- #
# Case processing
# --------------------------------------------------------------------------- #


def process_case(
    case: dict[str, Any],
    output_root: str,
    margin_mm: float,
    replace: bool = False,
    registration_spacing_mm: float = 4.0,
    bspline_grid_spacing_mm: float = 40.0,
    threads: int = 2,
) -> dict[str, Any]:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(max(1, threads))
    output_root_path = Path(output_root)
    case_dir = output_root_path / "cases" / case["case_id"]
    if output_is_complete(case_dir) and (not replace or output_has_current_pipeline(case_dir)):
        existing = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        state = "already_replaced" if replace else "already_complete"
        return {**existing, "run_state": state}

    case_dir.mkdir(parents=True, exist_ok=True)
    staging_dir: Path | None = None
    write_dir = case_dir
    if replace:
        staging_root = (output_root_path / ".staging").resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = (staging_root / f"{case['case_id']}-{os.getpid()}").resolve()
        if not staging_dir.is_relative_to(staging_root):
            raise ValueError(f"Unsafe staging path: {staging_dir}")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir()
        write_dir = staging_dir

    try:
        with tempfile.TemporaryDirectory(prefix=f"{case['case_id']}-") as temp_name:
            temp_dir = Path(temp_name)
            if case["dataset"] == "PLC-CECT" and not case.get("phase_geometry"):
                raise ValueError("PLC-CECT requires plc_geometry.csv (run estimate_plc_geometry.py first)")

            native: dict[str, sitk.Image] = {}
            image_read_details: dict[str, dict[str, Any]] = {}
            liver_masks: dict[str, sitk.Image] = {}
            for phase in PHASES:
                path = materialize(case["phase_sources"][phase], temp_dir, f"image-{phase}")
                image, details = read_scalar_image(path)
                image = apply_phase_geometry(image, case["phase_geometry"].get(phase))
                native[phase] = to_hounsfield(image, case["intensity_kind"])
                image_read_details[phase] = details
                liver_source = case["liver_sources"].get(phase)
                if liver_source is not None:
                    liver_path = materialize(liver_source, temp_dir, f"liver-{phase}")
                    liver_raw, _ = read_scalar_image(liver_path)
                    liver_raw = apply_phase_geometry(liver_raw, case["phase_geometry"].get(phase))
                    label = WAW_LIVER_LABEL if case["dataset"] == "WAW-TACE" else None
                    liver_masks[phase] = align_mask_to_reference(
                        binary_mask(liver_raw, label=label), native[phase]
                    )

            reference_phase = case["reference_phase"]
            if reference_phase not in liver_masks or not mask_has_foreground(liver_masks[reference_phase]):
                alternatives = [
                    phase
                    for phase in ("PVP", "AP", "DP", "NC")
                    if phase in liver_masks and mask_has_foreground(liver_masks[phase])
                ]
                if alternatives and case["dataset"] != "MCT-LTDiag":
                    reference_phase = alternatives[0]
            reference = native[reference_phase]
            if reference_phase in liver_masks:
                reference_liver = align_mask_to_reference(liver_masks[reference_phase], reference)
            else:
                reference_liver = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
                reference_liver.CopyInformation(reference)

            transforms: dict[str, sitk.Transform] = {
                reference_phase: sitk.Transform(3, sitk.sitkIdentity)
            }
            registrations: dict[str, dict[str, Any]] = {
                reference_phase: {
                    "method": "reference_phase",
                    "transform_file": None,
                    "final_liver_dice": 1.0,
                    "usable": True,
                }
            }
            for phase in PHASES:
                if phase == reference_phase:
                    continue
                if case["dataset"] == "MCT-LTDiag":
                    transforms[phase] = sitk.Transform(3, sitk.sitkIdentity)
                    registrations[phase] = {
                        "method": (
                            "released_registered_grid"
                            if same_grid(reference, native[phase])
                            else "released_physical_space_resampled_to_pvp_grid"
                        ),
                        "transform_file": None,
                        "final_liver_dice": None,
                        "usable": True,
                    }
                    continue
                moving_liver = liver_masks.get(phase)
                if moving_liver is None:
                    moving_liver = sitk.Image(native[phase].GetSize(), sitk.sitkUInt8)
                    moving_liver.CopyInformation(native[phase])
                transform, details = liver_guided_affine_bspline_register(
                    reference,
                    native[phase],
                    reference_liver,
                    moving_liver,
                    registration_spacing_mm,
                    bspline_grid_spacing_mm,
                )
                transforms[phase] = transform
                transform_name = f"transform_{phase.lower()}_to_{reference_phase.lower()}.h5"
                sitk.WriteTransform(transform, str(write_dir / transform_name))
                dice = details.get("final_liver_dice")
                registrations[phase] = {
                    **details,
                    "transform_file": transform_name,
                    "usable": bool(dice is not None and dice >= REGISTRATION_DICE_GATE),
                    "dice_gate": REGISTRATION_DICE_GATE,
                }

            if case["dataset"] == "WAW-TACE":
                tumour_inputs = [
                    (case["tumour_annotation_phase"], source)
                    for source in case["tumour_sources"]
                ]
            elif "tumour_sources_by_phase" in case:
                tumour_inputs = [
                    (reference_phase, source)
                    for source in case["tumour_sources_by_phase"].get(reference_phase, [])
                ]
            else:
                tumour_inputs = [
                    (reference_phase, source) for source in case["tumour_sources"]
                ]

            tumour_masks_native: list[tuple[str, sitk.Image]] = []
            for index, (source_phase, source) in enumerate(tumour_inputs):
                tumour_path = materialize(source, temp_dir, f"tumour-{index}")
                tumour_raw, _ = read_scalar_image(tumour_path)
                tumour_raw = apply_phase_geometry(tumour_raw, case["phase_geometry"].get(source_phase))
                tumour_masks_native.append(
                    (source_phase, align_mask_to_reference(binary_mask(tumour_raw), native[source_phase]))
                )

            # Tumour on the reference native grid, only to define the crop region.
            tumour_on_reference = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
            tumour_on_reference.CopyInformation(reference)
            for source_phase, mask in tumour_masks_native:
                warped = sitk.Resample(
                    mask, reference, transforms[source_phase], sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
                )
                warped.CopyInformation(reference)
                tumour_on_reference = sitk.Or(tumour_on_reference, warped)

            if mask_has_foreground(reference_liver):
                crop_mask = sitk.Or(reference_liver, tumour_on_reference)
                start, size = crop_region(crop_mask, margin_mm)
                crop_basis = "liver_union_tumour_plus_margin"
            elif mask_has_foreground(tumour_on_reference):
                start, size = crop_region(tumour_on_reference, margin_mm)
                crop_basis = "tumour_plus_margin_no_liver_mask"
            else:
                start, size = [0, 0, 0], list(reference.GetSize())
                crop_basis = "full_volume_no_masks"
            grid = isotropic_output_grid(reference, start, size)

            # One interpolation per channel: native grid -> transform -> 1 mm grid.
            outputs: dict[str, sitk.Image] = {}
            for phase in PHASES:
                outputs[phase] = resample_image_to_grid(native[phase], grid, transforms[phase])
            liver_out = resample_mask_to_grid(reference_liver, grid, transforms[reference_phase])
            tumour_out = sitk.Image(grid.GetSize(), sitk.sitkUInt8)
            tumour_out.CopyInformation(grid)
            for source_phase, mask in tumour_masks_native:
                warped = resample_mask_to_grid(mask, grid, transforms[source_phase])
                warped.CopyInformation(grid)
                tumour_out = sitk.Or(tumour_out, warped)

            for phase in PHASES:
                write_compressed_image(to_int16_hu(outputs[phase]), write_dir / OUTPUT_FILES[phase])
            write_compressed_image(liver_out, write_dir / "liver_mask.nii.gz")
            write_compressed_image(tumour_out, write_dir / "tumor_mask.nii.gz")

            tumour_voxels = int(sitk.GetArrayViewFromImage(tumour_out).sum())
            liver_voxels = int(sitk.GetArrayViewFromImage(liver_out).sum())
            liver_volume_ml = liver_voxels * math.prod(grid.GetSpacing()) / 1000.0
            annotation_phase = case.get("tumour_annotation_phase", reference_phase)
            propagation = registrations.get(annotation_phase, {})
            tumour_mask_reliable = bool(propagation.get("usable", True))
            sanity = sanity_checks(case, native, liver_masks, reference_phase, liver_volume_ml, registrations)
            usable_phases = [phase for phase in PHASES if registrations[phase].get("usable", True)]

            intensity_kind = case["intensity_kind"]
            record = {
                "status": "complete",
                "run_state": "replaced" if replace else "processed",
                "pipeline_version": PIPELINE_VERSION,
                "dataset": case["dataset"],
                "patient_id": case["patient_id"],
                "case_id": case["case_id"],
                "diagnosis": case["diagnosis"],
                "phase_order": list(PHASES),
                "reference_phase": reference_phase,
                "tumour_annotation_phase": annotation_phase,
                "tumour_mask_reliable": tumour_mask_reliable,
                "usable_phases": usable_phases,
                "registration_policy": case["registration_policy"],
                "registrations": registrations,
                "orientation": "LPS",
                "native_geometry": {phase: geometry(image) for phase, image in native.items()},
                "phase_geometry_source": case.get("phase_geometry", {}),
                "released_slice_thickness_mm": case.get("released_slice_thickness_mm"),
                "liver_mask_origin": case.get("liver_mask_origin", {}),
                "crop_basis": crop_basis,
                "crop_index_on_reference": start,
                "crop_size_on_reference": size,
                "crop_margin_mm": margin_mm,
                "output_geometry": geometry(grid),
                "output_dtype": {"images": "int16_hounsfield_units", "masks": "uint8_binary"},
                "resampling": {
                    "target_spacing_mm": list(TARGET_SPACING_MM),
                    "image_interpolation": "linear, single interpolation from the native grid through the phase transform",
                    "mask_interpolation": "nearest_neighbour, single interpolation",
                    "out_of_field_value_hu": OUT_OF_FIELD_HU,
                },
                "intensity_kind": intensity_kind,
                "intensity_source": (
                    "released_hounsfield_units"
                    if intensity_kind == "hu"
                    else "released_uint8_window_-200_200_HU_linearly_mapped_to_HU"
                ),
                "intensity_valid_range_hu": (
                    list(PLC_WINDOW_HU) if intensity_kind == "released_uint8_windowed" else None
                ),
                "intensity_note": (
                    "Values above 200 HU are saturated in the PLC-CECT release; quantisation step 1.57 HU."
                    if intensity_kind == "released_uint8_windowed"
                    else "Unclipped HU; clip in the data loader (e.g. [-200, 200]) for cross-dataset consistency."
                ),
                "image_read_details": image_read_details,
                "tumour_annotation_files": len(tumour_inputs),
                "tumour_voxels": tumour_voxels,
                "liver_voxels": liver_voxels,
                "liver_volume_ml": liver_volume_ml,
                "liver_mask_available": liver_voxels > 0,
                "sanity": sanity,
                "source_references": {
                    "images": case["phase_sources"],
                    "liver_masks": case["liver_sources"],
                    "tumour_masks": [source for _, source in tumour_inputs],
                },
            }
            write_json(write_dir / "case.json", record)

        if staging_dir is not None:
            for stale in list(case_dir.glob("transform_*.tfm")) + list(case_dir.glob("transform_*.h5")):
                stale.unlink()
            for staged_file in staging_dir.iterdir():
                os.replace(staged_file, case_dir / staged_file.name)
            staging_dir.rmdir()
        return record
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)


def failure_record(case: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "run_state": "failed",
        "dataset": case["dataset"],
        "patient_id": case["patient_id"],
        "case_id": case["case_id"],
        "diagnosis": case.get("diagnosis", ""),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    geometry_out = record.get("output_geometry", {})
    spacing = geometry_out.get("spacing_mm") or ["", "", ""]
    size = geometry_out.get("size") or ["", "", ""]
    native = record.get("native_geometry", {})
    reference_phase = record.get("reference_phase", "")
    reference_native = native.get(reference_phase, {}).get("spacing_mm") or ["", "", ""]
    registrations = record.get("registrations", {})
    phase_source = record.get("phase_geometry_source", {}) or {}
    sanity = record.get("sanity", {})

    def dice(phase: str) -> Any:
        value = registrations.get(phase, {}).get("final_liver_dice")
        return round(value, 4) if isinstance(value, (int, float)) else ""

    def usable(phase: str) -> Any:
        value = registrations.get(phase, {}).get("usable")
        return int(bool(value)) if value is not None else ""

    def native_z(phase: str) -> Any:
        value = (native.get(phase, {}).get("spacing_mm") or [None, None, None])[2]
        return round(value, 4) if isinstance(value, (int, float)) else ""

    confidences = {item.get("spacing_z_confidence") for item in phase_source.values()} - {None}
    confidence = "" if not confidences else ("low" if "low" in confidences else ("medium" if "medium" in confidences else "high"))
    return {
        "status": record["status"],
        "dataset": record["dataset"],
        "patient_id": record["patient_id"],
        "case_id": record["case_id"],
        "diagnosis": record.get("diagnosis", ""),
        "reference_phase": reference_phase,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "tumour_voxels": record.get("tumour_voxels", ""),
        "liver_voxels": record.get("liver_voxels", ""),
        "liver_volume_ml": round(record["liver_volume_ml"], 1) if "liver_volume_ml" in record else "",
        "native_spacing_x_mm": reference_native[0],
        "native_spacing_y_mm": reference_native[1],
        "native_spacing_z_mm": reference_native[2],
        "native_spacing_z_nc_mm": native_z("NC"),
        "native_spacing_z_ap_mm": native_z("AP"),
        "native_spacing_z_pvp_mm": native_z("PVP"),
        "native_spacing_z_dp_mm": native_z("DP"),
        "spacing_source": (
            "estimated_" + phase_source.get(reference_phase, {}).get("spacing_z_source", "")
            if phase_source
            else ("released_header" if record["status"] == "complete" else "")
        ),
        "spacing_confidence": confidence if phase_source else ("exact" if record["status"] == "complete" else ""),
        "intensity_source": record.get("intensity_source", ""),
        "intensity_valid_range_hu": (
            f"{record['intensity_valid_range_hu'][0]:.0f}..{record['intensity_valid_range_hu'][1]:.0f}"
            if record.get("intensity_valid_range_hu")
            else ("full" if record["status"] == "complete" else "")
        ),
        "registration_liver_dice_nc": dice("NC"),
        "registration_liver_dice_ap": dice("AP"),
        "registration_liver_dice_dp": dice("DP"),
        "usable_nc": usable("NC"),
        "usable_ap": usable("AP"),
        "usable_pvp": usable("PVP"),
        "usable_dp": usable("DP"),
        "usable_phases": "+".join(record.get("usable_phases", [])),
        "tumour_annotation_phase": record.get("tumour_annotation_phase", ""),
        "tumour_mask_reliable": int(bool(record["tumour_mask_reliable"])) if "tumour_mask_reliable" in record else "",
        "sanity_pass": int(bool(sanity.get("pass"))) if sanity else "",
        "sanity_flags": ";".join(sanity.get("flags", [])) if sanity else "",
        "error": record.get("error", ""),
    }


def collect_completed_records(output_root: Path) -> dict[str, dict[str, Any]]:
    """Every case.json under cases/ that carries the current pipeline version."""
    records: dict[str, dict[str, Any]] = {}
    cases_root = output_root / "cases"
    if not cases_root.is_dir():
        return records
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if not output_has_current_pipeline(case_dir):
            continue
        record = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        records[record["case_id"]] = record
    return records


def write_manifest(output_root: Path, run_records: list[dict[str, Any]]) -> None:
    # The manifest describes the whole output root, not only this invocation:
    # cases completed by earlier invocations (other datasets, resumed runs) are
    # merged from their case.json; failures come from this invocation.
    merged = collect_completed_records(output_root)
    for record in run_records:
        if record["status"] != "complete":
            merged[record["case_id"]] = record
    records = sorted(merged.values(), key=lambda record: (record["dataset"], str(record["patient_id"])))
    flat_rows = [manifest_row(record) for record in records]
    with (output_root / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    failures = [record for record in records if record["status"] != "complete"]
    write_json(output_root / "failures.json", failures)
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {"complete": 0, "failed": 0})
    usable_counts: dict[str, dict[str, int]] = defaultdict(lambda: {phase: 0 for phase in PHASES})
    sanity_fail: dict[str, int] = defaultdict(int)
    for record in records:
        by_dataset[record["dataset"]][record["status"]] += 1
        for phase in record.get("usable_phases", []):
            usable_counts[record["dataset"]][phase] += 1
        if record.get("sanity") and not record["sanity"]["pass"]:
            sanity_fail[record["dataset"]] += 1
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "phase_order": list(PHASES),
        "orientation": "LPS",
        "image_dtype": "int16 Hounsfield units; -1000 outside the acquired field of view",
        "intensity_policy": (
            "No clipping or normalisation in preprocessing. PLC-CECT is limited to "
            "[-200, 200] HU by its 8-bit release; clip all datasets to at most that range in the loader."
        ),
        "spacing_policy": "all cases resampled once to 1 x 1 x 1 mm on a liver-centred grid",
        "plc_geometry_policy": (
            "in-plane 0.78125 mm from the 400 mm / 512 protocol; per-phase slice spacing "
            "estimated from vertebra centroid distances (estimate_plc_geometry.py)"
        ),
        "registration_dice_gate": REGISTRATION_DICE_GATE,
        "case_counts": dict(by_dataset),
        "usable_phase_counts": {k: dict(v) for k, v in usable_counts.items()},
        "sanity_failures": dict(sanity_fail),
        "total_complete": sum(record["status"] == "complete" for record in records),
        "total_failed": len(failures),
    }
    write_json(output_root / "summary.json", summary)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.crop_margin_mm < 0:
        raise SystemExit("--crop-margin-mm must be non-negative")
    if args.registration_spacing_mm <= 0:
        raise SystemExit("--registration-spacing-mm must be positive")
    if args.bspline_grid_spacing_mm <= 0:
        raise SystemExit("--bspline-grid-spacing-mm must be positive")
    unselected_replacements = set(args.replace_datasets) - set(args.datasets)
    if unselected_replacements:
        raise SystemExit(
            "--replace-datasets must be a subset of --datasets: "
            + ", ".join(sorted(unselected_replacements))
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    waw_selection = args.waw_selection or (args.output_root / "manifests" / "waw_four_phase_patients.csv")
    plc_geometry = args.plc_geometry or (args.output_root / "manifests" / "plc_geometry.csv")
    if "PLC-CECT" in args.datasets and not plc_geometry.is_file():
        raise SystemExit(f"PLC geometry file not found: {plc_geometry} (run estimate_plc_geometry.py)")
    builders = {
        "PLC-CECT": lambda: build_plc_cases(
            resolve_dataset_root("PLC-CECT", args.raw_root, args.plc_root),
            plc_geometry,
            set(args.patients) if args.patients else None,
        ),
        "MCT-LTDiag": lambda: build_mct_cases(resolve_dataset_root("MCT-LTDiag", args.raw_root, args.mct_root)),
        "WAW-TACE": lambda: build_waw_cases(
            resolve_dataset_root("WAW-TACE", args.raw_root, args.waw_root), waw_selection
        ),
    }
    cases: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_cases = builders[dataset]()
        if args.patients:
            wanted = set(args.patients)
            dataset_cases = [case for case in dataset_cases if case["patient_id"] in wanted]
        if args.limit_per_dataset is not None:
            dataset_cases = dataset_cases[: args.limit_per_dataset]
        cases.extend(dataset_cases)
    print(f"Prepared {len(cases)} case specifications", flush=True)

    records: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, case in enumerate(cases, start=1):
            try:
                record = process_case(
                    case,
                    str(args.output_root),
                    args.crop_margin_mm,
                    replace=case["dataset"] in args.replace_datasets,
                    registration_spacing_mm=args.registration_spacing_mm,
                    bspline_grid_spacing_mm=args.bspline_grid_spacing_mm,
                    threads=args.threads_per_worker,
                )
                print(
                    f"[{index}/{len(cases)}] {case['case_id']}: {record['run_state']}",
                    flush=True,
                )
            except Exception as exc:  # continue to produce an exact failure inventory
                record = failure_record(case, exc)
                print(f"[{index}/{len(cases)}] {case['case_id']}: FAILED: {exc}", flush=True)
            records.append(record)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_case,
                    case,
                    str(args.output_root),
                    args.crop_margin_mm,
                    case["dataset"] in args.replace_datasets,
                    args.registration_spacing_mm,
                    args.bspline_grid_spacing_mm,
                    args.threads_per_worker,
                ): case
                for case in cases
            }
            for index, future in enumerate(as_completed(futures), start=1):
                case = futures[future]
                try:
                    record = future.result()
                    print(
                        f"[{index}/{len(cases)}] {case['case_id']}: {record['run_state']}",
                        flush=True,
                    )
                except Exception as exc:
                    record = failure_record(case, exc)
                    print(f"[{index}/{len(cases)}] {case['case_id']}: FAILED: {exc}", flush=True)
                records.append(record)
    records.sort(key=lambda record: (record["dataset"], str(record["patient_id"])))
    write_manifest(args.output_root, records)
    failures = sum(record["status"] != "complete" for record in records)
    print(f"Complete: {len(records) - failures}; failed: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
