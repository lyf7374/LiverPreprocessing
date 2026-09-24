"""Estimate the physical voxel geometry of the released PLC-CECT volumes.

The PLC-CECT release stores every CT as uint8 with a 1 x 1 x 1 mm header, a
zero origin and a header that declares the slice axis k -> superior.  Three
things are wrong or missing in that header and are recovered here:

* In-plane spacing.  Fixed by the acquisition protocol (512 x 512 matrix,
  400 mm display field of view -> 0.78125 mm).  Confirmed by the liver width.
* Slice order.  Most volumes are stored superior -> inferior (the liver dome
  is at k = 0), a minority inferior -> superior, occasionally differing between
  phases of one patient.  Decided per volume from the liver-mask area profile
  (the dome end tapers abruptly, the inferior tip slowly), cross-checked
  between phases by profile correlation and verified with the vertebra order
  returned by TotalSegmentator.
* Slice spacing.  Differs between patients and between the four phases of one
  patient.  Estimated per volume from the distances between consecutive
  vertebra centroids segmented by the TotalSegmentator 3 mm model (run
  in-process through the nnU-Net predictor), compared with level-specific adult
  reference distances, then fused per patient through the exact constraint
  that the liver spans the same physical z range in every phase.

``measure`` writes one JSON record per volume (resumable); ``solve`` writes
``plc_geometry.csv`` with spacing, slice-order flip, provenance and a
confidence grade per volume.  The expected error of the slice spacing is
about +/-10 %; it is recorded so that evaluation can stratify by confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPENDENCIES = PROJECT_ROOT / "tmp" / "python_deps"
if LOCAL_DEPENDENCIES.exists():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import SimpleITK as sitk

from preprocess_multiphase_ct import (
    PLC_IN_PLANE_SPACING_MM,
    PLC_PHASES,
    PHASES,
    build_plc_cases,
    flip_k_axis,
    resolve_dataset_root,
    materialize,
    plc_uint8_to_hu,
    read_scalar_image,
)

GEOMETRY_VERSION = "plc_geometry_v2_totalsegmentator3mm_2026-09-24"
LIVER_Z_EXTENT_PRIOR_MM = 170.0
MODEL_SPACING_MM = 3.0
MIN_VERTEBRA_VOXELS_3MM = 120  # ~3.2 mL; a complete thoracolumbar vertebra is > 15 mL
MIN_LIVER_PROFILE_SLICES = 8
MIN_RELEASED_LIVER_FRACTION = 0.4  # of the volume's slices; the release is liver-cropped
RELATIVE_FLIP_CORRELATION_MARGIN = 0.05
PATIENT_ORIENTATION_MARGIN = 0.30

# Adult reference distance (mm) between the centroids of consecutive vertebrae
# (upper -> lower): mean vertebral-body height plus intervertebral disc.
VERTEBRA_PAIR_REFERENCE_MM = {
    ("T7", "T8"): 23.5,
    ("T8", "T9"): 24.5,
    ("T9", "T10"): 25.5,
    ("T10", "T11"): 27.0,
    ("T11", "T12"): 29.0,
    ("T12", "L1"): 31.5,
    ("L1", "L2"): 33.5,
    ("L2", "L3"): 35.0,
    ("L3", "L4"): 35.5,
    ("L4", "L5"): 35.0,
}
VERTEBRA_ORDER = ["T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
PLAUSIBLE_LIVER_LEVELS = {"T9", "T10", "T11", "T12", "L1", "L2", "L3"}
GENERIC_PAIR_REFERENCE_MM = 30.0
CONFIRMED_ORDER_SCORE = 4
PHASE_EXTENT_INCONSISTENCY = 0.15  # relative disagreement that overrides the shared liver extent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-weights", help="Fetch the TotalSegmentator 3 mm weights (task 297) once.")
    for name in ("measure", "solve", "all"):
        p = sub.add_parser(name)
        p.add_argument("--raw-root", type=Path, default=None, help="Folder holding the PLC-CECT release folder.")
        p.add_argument("--plc-root", type=Path, default=None, help="PLC-CECT release folder (default <raw-root>/PLC-CECT).")
        p.add_argument("--output-root", type=Path, required=True, help="Pipeline output root; writes manifests/plc_geometry*.")
        p.add_argument("--limit", type=int, default=None, help="Only the first N patients.")
        p.add_argument("--patients", nargs="*", default=None, help="Only these patient ids.")
        p.add_argument("--force", action="store_true", help="Re-measure even if a record exists.")
        p.add_argument(
            "--reference-scale",
            type=float,
            default=1.0,
            help="Multiply the vertebra reference distances (population calibration).",
        )
        p.add_argument("--device", default="cuda")
    return parser.parse_args()


def geometry_dir(output_root: Path) -> Path:
    return output_root / "manifests" / "plc_geometry"


def measurements_path(output_root: Path) -> Path:
    return geometry_dir(output_root) / "plc_vertebra_measurements.jsonl"


def geometry_csv_path(output_root: Path) -> Path:
    return output_root / "manifests" / "plc_geometry.csv"


# --------------------------------------------------------------------------- #
# TotalSegmentator in-process
# --------------------------------------------------------------------------- #


MODEL_TASK_ID = 297
MODEL_FOLDER = "Dataset297_TotalSegmentator_total_3mm_1559subj"
MODEL_TRAINER = "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"


def weights_dir() -> Path:
    """TotalSegmentator's weight store: $TOTALSEG_WEIGHTS_PATH or ~/.totalsegmentator/nnunet/results."""
    override = os.environ.get("TOTALSEG_WEIGHTS_PATH")
    if override:
        return Path(override)
    return Path.home() / ".totalsegmentator" / "nnunet" / "results"


def download_weights() -> None:
    from totalsegmentator.libs import download_pretrained_weights

    download_pretrained_weights(MODEL_TASK_ID)
    model_dir = weights_dir() / MODEL_FOLDER / MODEL_TRAINER
    if not (model_dir / "fold_0" / "checkpoint_final.pth").is_file():
        raise SystemExit(f"Download finished but weights are not at {model_dir}")
    print(f"TotalSegmentator 3 mm weights ready: {model_dir}")


def load_predictor(device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = weights_dir() / MODEL_FOLDER / MODEL_TRAINER
    if not (model_dir / "fold_0" / "checkpoint_final.pth").is_file():
        raise SystemExit(
            f"TotalSegmentator 3 mm weights not found at {model_dir}. Run "
            "'estimate_plc_geometry.py download-weights' once (or set TOTALSEG_WEIGHTS_PATH)."
        )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    return predictor


def class_names() -> dict[int, str]:
    from totalsegmentator.map_to_binary import class_map

    return {int(k): v for k, v in class_map["total"].items()}


def resample_to_model_grid(image: sitk.Image, spacing: float = MODEL_SPACING_MM) -> sitk.Image:
    """RAS voxel order (what the TotalSegmentator nnU-Net models were trained on)."""
    ras = sitk.DICOMOrient(image, "RAS")
    size = [max(1, int(math.ceil(n * s / spacing))) for n, s in zip(ras.GetSize(), ras.GetSpacing())]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing((spacing, spacing, spacing))
    grid.SetOrigin(ras.GetOrigin())
    grid.SetDirection(ras.GetDirection())
    return sitk.Resample(
        sitk.Cast(ras, sitk.sitkFloat32),
        grid,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )


def segment(predictor, hu_image: sitk.Image) -> tuple[np.ndarray, sitk.Image]:
    model_image = resample_to_model_grid(hu_image)
    data = sitk.GetArrayFromImage(model_image)[None].astype(np.float32)
    spacing = list(model_image.GetSpacing())[::-1]
    seg = predictor.predict_single_npy_array(data, {"spacing": spacing})
    return np.asarray(seg, dtype=np.uint8), model_image


# --------------------------------------------------------------------------- #
# Measurements on a segmentation
# --------------------------------------------------------------------------- #


def mask_z_profile(mask: np.ndarray) -> dict[str, Any]:
    """mask is (z, y, x)."""
    per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
    nz = np.flatnonzero(per_slice)
    if nz.size == 0:
        return {"voxels": 0, "z0": None, "z1": None, "z_extent": 0}
    return {
        "voxels": int(per_slice.sum()),
        "z0": int(nz[0]),
        "z1": int(nz[-1]),
        "z_extent": int(nz[-1] - nz[0] + 1),
    }


def vertebra_records(seg: np.ndarray, names: dict[int, str]) -> list[dict[str, Any]]:
    """Per detected vertebra: level, centroid z and extent in model-grid mm, sorted inferior -> superior."""
    records = []
    nz_total = seg.shape[0]
    z_index = np.arange(nz_total)
    for label in np.unique(seg):
        name = names.get(int(label), "")
        if not name.startswith("vertebrae_"):
            continue
        level = name.split("_", 1)[1]
        if level not in VERTEBRA_ORDER:
            continue
        mask = seg == label
        voxels = int(mask.sum())
        if voxels < MIN_VERTEBRA_VOXELS_3MM:
            continue
        z_profile = mask.reshape(nz_total, -1).sum(axis=1)
        zs = np.flatnonzero(z_profile)
        records.append(
            {
                "level": level,
                "order": VERTEBRA_ORDER.index(level),
                "voxels_3mm": voxels,
                "volume_ml_at_guess": voxels * MODEL_SPACING_MM**3 / 1000.0,
                "centroid_z_mm_at_guess": float((z_profile * z_index).sum() / z_profile.sum()) * MODEL_SPACING_MM,
                "z0_mm_at_guess": float(zs[0]) * MODEL_SPACING_MM,
                "z1_mm_at_guess": float(zs[-1] + 1) * MODEL_SPACING_MM,
                "touches_volume_edge": bool(zs[0] == 0 or zs[-1] == nz_total - 1),
            }
        )
    records.sort(key=lambda r: r["centroid_z_mm_at_guess"])
    return records


def vertebra_order_score(vertebrae: list[dict[str, Any]]) -> int:
    """Kendall-style order score of the levels sorted along z.

    +1 for every pair of vertebrae whose levels ascend with z (lumbar below
    thoracic), -1 for every inverted pair.  A clean run of four consecutive
    levels scores 6; an upside-down or scrambled volume scores <= 0.
    """
    score = 0
    for index, lower in enumerate(vertebrae):
        for upper in vertebrae[index + 1 :]:
            if upper["order"] < lower["order"]:
                score += 1
            elif upper["order"] > lower["order"]:
                score -= 1
    return score


def plausible_level_count(vertebrae: list[dict[str, Any]]) -> int:
    """Vertebrae at levels a liver scan normally covers (T9-L3)."""
    return sum(v["level"] in PLAUSIBLE_LIVER_LEVELS for v in vertebrae)


def organ_records(seg: np.ndarray, names: dict[int, str]) -> dict[str, Any]:
    wanted = {"liver", "spleen", "kidney_left", "kidney_right"}
    out: dict[str, Any] = {}
    nz = seg.shape[0]
    for label, name in names.items():
        if name not in wanted:
            continue
        profile = mask_z_profile(seg == label)
        out[name] = {
            "voxels_3mm": profile["voxels"],
            "volume_ml_at_guess": profile["voxels"] * MODEL_SPACING_MM**3 / 1000.0,
            "z_extent_mm_at_guess": profile["z_extent"] * MODEL_SPACING_MM,
            "touches_volume_edge": bool(
                profile["voxels"] > 0 and (profile["z0"] == 0 or profile["z1"] == nz - 1)
            ),
        }
    return out


# --------------------------------------------------------------------------- #
# Slice-order decision from liver area profiles
# --------------------------------------------------------------------------- #


def liver_area_profile(mask_array: np.ndarray) -> np.ndarray:
    """Liver voxels per slice along k, trimmed to the liver extent (empty if none)."""
    per_slice = mask_array.reshape(mask_array.shape[0], -1).sum(axis=1).astype(float)
    nz = np.flatnonzero(per_slice)
    if nz.size == 0:
        return np.zeros(0)
    return per_slice[nz[0] : nz[-1] + 1]


def dome_margin(profile: np.ndarray) -> float:
    """Positive when the abruptly tapering (dome) end of the liver is at low k.

    Calibrated on MCT-LTDiag and WAW-TACE, whose headers are correct: with
    k -> superior the dome is at high k and the margin is negative in 80/80
    cases.  A positive margin therefore means the slices are stored
    superior-first and must be reversed.
    """
    n = len(profile)
    if n < MIN_LIVER_PROFILE_SLICES:
        return 0.0
    w = max(2, n // 5)
    peak = max(1.0, float(profile.max()))
    rise_low = (profile[w] - profile[0]) / peak
    rise_high = (profile[-1 - w] - profile[-1]) / peak
    return float(rise_low - rise_high)


def profile_correlations(profile: np.ndarray, reference: np.ndarray, samples: int = 100) -> tuple[float, float]:
    def resample(p: np.ndarray) -> np.ndarray:
        return np.interp(np.linspace(0.0, 1.0, samples), np.linspace(0.0, 1.0, len(p)), p)

    a, b = resample(profile), resample(reference)
    if a.std() == 0 or b.std() == 0:
        return 0.0, 0.0
    direct = float(np.corrcoef(a, b)[0, 1])
    reversed_ = float(np.corrcoef(a[::-1], b)[0, 1])
    return direct, reversed_


def decide_orientation(profiles: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Per phase: whether the stored slice order must be reversed."""
    usable = {p: prof for p, prof in profiles.items() if len(prof) >= MIN_LIVER_PROFILE_SLICES}
    result: dict[str, dict[str, Any]] = {}
    if not usable:
        for phase in profiles:
            result[phase] = {
                "flip": True,
                "source": "dataset_default_no_liver_profile",
                "margin": 0.0,
                "relative_flip": False,
                "correlation_direct": None,
                "correlation_reversed": None,
                "patient_evidence": 0.0,
            }
        return result
    reference = "PVP" if "PVP" in usable else max(usable, key=lambda p: len(usable[p]))
    relative: dict[str, tuple[bool, float | None, float | None]] = {}
    for phase, profile in usable.items():
        if phase == reference:
            relative[phase] = (False, None, None)
            continue
        direct, reversed_ = profile_correlations(profile, usable[reference])
        relative[phase] = (reversed_ > direct + RELATIVE_FLIP_CORRELATION_MARGIN, direct, reversed_)
    evidence = 0.0
    for phase, profile in usable.items():
        margin = dome_margin(profile)
        evidence += -margin if relative[phase][0] else margin
    reference_flip = evidence >= 0.0  # dome at low k in the reference orientation
    if abs(evidence) >= PATIENT_ORIENTATION_MARGIN:
        source = "liver_profile"
    else:
        source = "liver_profile_weak"
    for phase in profiles:
        if phase in usable:
            rel_flip, direct, reversed_ = relative[phase]
            margin = dome_margin(usable[phase])
        else:
            rel_flip, direct, reversed_ = False, None, None
            margin = 0.0
        result[phase] = {
            "flip": bool(reference_flip != rel_flip),
            "source": source,
            "margin": round(margin, 3),
            "relative_flip": bool(rel_flip),
            "correlation_direct": None if direct is None else round(direct, 3),
            "correlation_reversed": None if reversed_ is None else round(reversed_, 3),
            "patient_evidence": round(evidence, 3),
        }
    return result


# --------------------------------------------------------------------------- #
# Measurement per patient
# --------------------------------------------------------------------------- #


def read_phase(case: dict[str, Any], phase: str, temp_dir: Path) -> tuple[sitk.Image, sitk.Image, dict[str, Any]]:
    image_path = materialize(case["phase_sources"][phase], temp_dir, f"image-{phase}")
    image, read_details = read_scalar_image(image_path)
    liver_path = materialize(case["liver_sources"][phase], temp_dir, f"liver-{phase}")
    liver_raw, _ = read_scalar_image(liver_path)
    liver = sitk.Cast(liver_raw > 0, sitk.sitkUInt8)
    if liver.GetSize() != image.GetSize():
        raise ValueError(f"{case['patient_id']} {phase}: liver mask size {liver.GetSize()} != CT {image.GetSize()}")
    liver.CopyInformation(image)
    return image, liver, read_details


def write_liver_fallback(
    seg: np.ndarray,
    liver_label: int,
    model_image: sitk.Image,
    oriented_hu: sitk.Image,
    flipped: bool,
    stored_reference: sitk.Image,
    path: Path,
) -> dict[str, Any]:
    """TotalSegmentator liver written back on the stored (unflipped) native grid."""
    seg_image = sitk.GetImageFromArray((seg == liver_label).astype(np.uint8))
    seg_image.CopyInformation(model_image)
    native = sitk.Resample(
        seg_image, oriented_hu, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    native = sitk.DICOMOrient(native, "LPS")
    array = sitk.GetArrayFromImage(native)
    if flipped:
        array = array[::-1]
    stored = sitk.GetImageFromArray(np.ascontiguousarray(array))
    stored.CopyInformation(stored_reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(True)
    writer.Execute(stored)
    return mask_z_profile(array > 0)


def measure_patient(
    predictor,
    names: dict[int, str],
    case: dict[str, Any],
    temp_dir: Path,
    liver_fallback_dir: Path,
) -> list[dict[str, Any]]:
    started = time.time()
    stored: dict[str, tuple[sitk.Image, sitk.Image, dict[str, Any]]] = {}
    profiles: dict[str, np.ndarray] = {}
    for phase in PHASES:
        image, liver, details = read_phase(case, phase, temp_dir)
        stored[phase] = (image, liver, details)
        profiles[phase] = liver_area_profile(sitk.GetArrayViewFromImage(liver) > 0)
    orientation = decide_orientation(profiles)
    liver_label = next(k for k, v in names.items() if v == "liver")

    records = []
    for phase in PHASES:
        image, liver, details = stored[phase]
        liver_profile = mask_z_profile(sitk.GetArrayViewFromImage(liver) > 0)
        size = list(image.GetSize())
        volume_guess = LIVER_Z_EXTENT_PRIOR_MM / max(1, int(round(size[2] * 0.96)))
        if liver_profile["z_extent"] >= MIN_LIVER_PROFILE_SLICES:
            guess = LIVER_Z_EXTENT_PRIOR_MM / liver_profile["z_extent"]
            guess_source = "released_liver_mask_extent"
        else:
            guess = volume_guess
            guess_source = "volume_slice_count"
        guess = float(min(7.0, max(0.4, guess)))
        volume_guess = float(min(7.0, max(0.4, volume_guess)))
        flip = orientation[phase]["flip"]
        hu = plc_uint8_to_hu(image)
        hu.SetSpacing((PLC_IN_PLANE_SPACING_MM, PLC_IN_PLANE_SPACING_MM, guess))
        oriented = flip_k_axis(hu) if flip else hu
        seg, model_image = segment(predictor, oriented)
        if (
            guess_source == "released_liver_mask_extent"
            and liver_profile["z_extent"] < MIN_RELEASED_LIVER_FRACTION * size[2]
            and abs(math.log(volume_guess / guess)) > math.log(1.5)
        ):
            # A released mask spanning a small part of a liver-cropped volume is
            # usually a failed segmentation whose guess scales the model input
            # badly.  Segment again with the volume-based guess and keep the pass
            # whose vertebra labels are more consistent.
            hu_alt = plc_uint8_to_hu(image)
            hu_alt.SetSpacing((PLC_IN_PLANE_SPACING_MM, PLC_IN_PLANE_SPACING_MM, volume_guess))
            oriented_alt = flip_k_axis(hu_alt) if flip else hu_alt
            seg_alt, model_alt = segment(predictor, oriented_alt)
            first = vertebra_records(seg, names)
            second = vertebra_records(seg_alt, names)
            if (vertebra_order_score(second), plausible_level_count(second), len(second)) > (
                vertebra_order_score(first), plausible_level_count(first), len(first)
            ):
                guess, guess_source = volume_guess, "volume_slice_count_after_failed_mask"
                hu, oriented, seg, model_image = hu_alt, oriented_alt, seg_alt, model_alt
        vertebrae = vertebra_records(seg, names)
        order_score = vertebra_order_score(vertebrae)
        plausible = plausible_level_count(vertebrae)
        orientation_source = orientation[phase]["source"]
        alternative: dict[str, Any] | None = None
        if order_score < CONFIRMED_ORDER_SCORE:
            # The vertebra labels do not clearly confirm the profile decision
            # (a correctly oriented liver scan yields >= 4 consecutive levels in
            # order; an upside-down one yields scrambled, implausible levels):
            # segment the other slice order too and keep the more consistent one.
            other = hu if flip else flip_k_axis(hu)
            seg_other, model_other = segment(predictor, other)
            vertebrae_other = vertebra_records(seg_other, names)
            score_other = vertebra_order_score(vertebrae_other)
            plausible_other = plausible_level_count(vertebrae_other)
            alternative = {
                "order_score": score_other,
                "plausible_levels": plausible_other,
                "levels": [v["level"] for v in vertebrae_other],
            }
            if (score_other, plausible_other) > (order_score, plausible):
                flip = not flip
                oriented, seg, model_image = other, seg_other, model_other
                vertebrae, order_score, plausible = vertebrae_other, score_other, plausible_other
                orientation_source = "totalsegmentator_vertebra_order_override"
            elif order_score > 0:
                orientation_source = f"{orientation_source}+vertebra_order_weakly_confirmed"
            else:
                orientation_source = f"{orientation_source}+vertebra_order_unconfirmed"
        else:
            orientation_source = f"{orientation_source}+vertebra_order_confirmed"
        organs = organ_records(seg, names)

        fallback_file = None
        fallback_profile = None
        ts_liver_slices = organs["liver"]["z_extent_mm_at_guess"] / guess if organs["liver"]["voxels_3mm"] else 0.0
        released_small = (
            liver_profile["z_extent"] < MIN_LIVER_PROFILE_SLICES
            or (ts_liver_slices > 0 and liver_profile["z_extent"] < 0.6 * ts_liver_slices)
        )
        if released_small and organs["liver"]["voxels_3mm"] > 0:
            fallback_file = liver_fallback_dir / f"{case['patient_id']}_{phase}_liver_totalsegmentator.nii.gz"
            fallback_profile = write_liver_fallback(
                seg, liver_label, model_image, oriented, flip, image, fallback_file
            )

        records.append(
            {
                "patient_id": case["patient_id"],
                "phase": phase,
                "geometry_version": GEOMETRY_VERSION,
                "size": size,
                "vector_fix": details.get("vector_fix"),
                "released_liver_slices": liver_profile["z_extent"],
                "released_liver_voxels": liver_profile["voxels"],
                "released_liver_small": bool(released_small),
                "k_axis_flip": bool(flip),
                "orientation": {**orientation[phase], "flip": bool(flip), "source": orientation_source},
                "vertebra_order_score": order_score,
                "vertebra_order_alternative": alternative,
                "slice_spacing_guess_mm": guess,
                "slice_spacing_guess_source": guess_source,
                "model_grid_size": list(model_image.GetSize()),
                "vertebrae": vertebrae,
                "organs": organs,
                "totalsegmentator_liver_native_file": str(fallback_file) if fallback_file else None,
                "totalsegmentator_liver_native_slices": fallback_profile["z_extent"] if fallback_profile else None,
                "seconds": round((time.time() - started) / len(PHASES), 1),
            }
        )
    return records


def load_measurements(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("geometry_version") == GEOMETRY_VERSION and "error" not in record:
                done[(record["patient_id"], record["phase"])] = record
    return done


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = resolve_dataset_root("PLC-CECT", args.raw_root, args.plc_root)
    cases = build_plc_cases(root, geometry_csv=None)
    if args.patients:
        wanted = set(args.patients)
        cases = [case for case in cases if case["patient_id"] in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def run_measure(args: argparse.Namespace) -> None:
    cases = select_cases(args)
    path = measurements_path(args.output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_measurements(path)
    jobs = [
        case
        for case in cases
        if args.force or any((case["patient_id"], phase) not in done for phase in PHASES)
    ]
    print(f"PLC patients: {len(cases)}; already measured: {len(cases) - len(jobs)}; to do: {len(jobs)}", flush=True)
    if not jobs:
        return
    predictor = load_predictor(args.device)
    names = class_names()
    liver_fallback_dir = geometry_dir(args.output_root) / "liver_fallback"
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
    started = time.time()
    with path.open("a", encoding="utf-8") as handle, tempfile.TemporaryDirectory(prefix="plc-geometry-") as temp:
        temp_dir = Path(temp)
        for index, case in enumerate(jobs, start=1):
            try:
                records = measure_patient(predictor, names, case, temp_dir, liver_fallback_dir)
            except Exception as exc:  # keep an exact inventory of failures
                records = [
                    {
                        "patient_id": case["patient_id"],
                        "phase": phase,
                        "geometry_version": GEOMETRY_VERSION,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for phase in PHASES
                ]
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(jobs):
                elapsed = time.time() - started
                print(
                    f"[{index}/{len(jobs)}] {case['patient_id']}: "
                    f"{records[0].get('seconds', 'failed')} s/volume; elapsed {elapsed / 60:.1f} min",
                    flush=True,
                )


# --------------------------------------------------------------------------- #
# Solve
# --------------------------------------------------------------------------- #


def vertebra_estimate(record: dict[str, Any], reference_scale: float) -> dict[str, Any]:
    """Absolute slice spacing from consecutive-vertebra centroid distances."""
    guess = record["slice_spacing_guess_mm"]
    vertebrae = [v for v in record.get("vertebrae", []) if not v["touches_volume_edge"]]
    pairs = []
    for lower, upper in zip(vertebrae, vertebrae[1:]):
        delta_mm_at_guess = upper["centroid_z_mm_at_guess"] - lower["centroid_z_mm_at_guess"]
        delta_slices = delta_mm_at_guess / guess
        if delta_slices <= 0:
            continue
        adjacent = upper["order"] + 1 == lower["order"]
        key = (upper["level"], lower["level"])
        if adjacent and key in VERTEBRA_PAIR_REFERENCE_MM:
            reference = VERTEBRA_PAIR_REFERENCE_MM[key] * reference_scale
            kind = "level_specific"
        elif upper["order"] < lower["order"] and 0.5 * GENERIC_PAIR_REFERENCE_MM <= delta_mm_at_guess <= 1.6 * GENERIC_PAIR_REFERENCE_MM:
            reference = GENERIC_PAIR_REFERENCE_MM * reference_scale
            kind = "generic"
        else:
            continue
        pairs.append(
            {
                "upper": upper["level"],
                "lower": lower["level"],
                "delta_slices": delta_slices,
                "reference_mm": reference,
                "kind": kind,
                "spacing_mm": reference / delta_slices,
            }
        )
    specific = [p for p in pairs if p["kind"] == "level_specific"]
    used = specific if specific else pairs
    if not used:
        return {"spacing_mm": None, "pairs": pairs, "n_pairs": 0, "kind": None}
    spacing = sum(p["reference_mm"] for p in used) / sum(p["delta_slices"] for p in used)
    return {
        "spacing_mm": spacing,
        "pairs": pairs,
        "n_pairs": len(used),
        "kind": "level_specific" if used is specific else "generic",
    }


def weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2.0:
            return value
    return ordered[-1][0]


def relative_fallback_path(value: str | None, output_root: Path | None) -> str:
    """Store fallback liver files relative to manifests/ so the CSV is portable."""
    if not value:
        return ""
    path = Path(value)
    if output_root is not None:
        try:
            return path.resolve().relative_to((output_root / "manifests").resolve()).as_posix()
        except ValueError:
            pass
    return f"plc_geometry/liver_fallback/{path.name}"


def solve_patient(
    records: dict[str, dict[str, Any]], reference_scale: float, output_root: Path | None = None
) -> list[dict[str, Any]]:
    """Fuse per-phase estimates through the shared liver z extent."""
    per_phase: dict[str, dict[str, Any]] = {}
    for phase, record in records.items():
        if record["released_liver_small"] and record.get("totalsegmentator_liver_native_slices"):
            liver_slices = record["totalsegmentator_liver_native_slices"]
            liver_source = "totalsegmentator"
        elif record["released_liver_slices"] >= MIN_LIVER_PROFILE_SLICES:
            liver_slices = record["released_liver_slices"]
            liver_source = "released"
        else:
            liver_slices = 0
            liver_source = "none"
        per_phase[phase] = {
            "record": record,
            "liver_slices": liver_slices,
            "liver_source": liver_source,
            "vertebra": vertebra_estimate(record, reference_scale),
        }
    extents: list[tuple[float, float]] = []
    extent_by_phase: dict[str, float] = {}
    for phase, item in per_phase.items():
        spacing = item["vertebra"]["spacing_mm"]
        if spacing and item["liver_slices"]:
            weight = item["vertebra"]["n_pairs"] * (2.0 if item["vertebra"]["kind"] == "level_specific" else 1.0)
            extents.append((spacing * item["liver_slices"], weight))
            extent_by_phase[phase] = spacing * item["liver_slices"]
    if extents:
        liver_extent = weighted_median(extents)
        # Phases whose own estimate contradicts the shared extent keep their own
        # value below; the agreement of the remaining phases defines the confidence.
        consistent = [
            value
            for phase, value in extent_by_phase.items()
            if not (
                per_phase[phase]["vertebra"]["kind"] == "level_specific"
                and per_phase[phase]["vertebra"]["n_pairs"] >= 2
                and abs(math.log(value / liver_extent)) > math.log(1.0 + PHASE_EXTENT_INCONSISTENCY)
            )
        ] or [liver_extent]
        spread = (max(consistent) - min(consistent)) / liver_extent if len(consistent) > 1 else 0.0
        extent_source = "vertebrae"
    else:
        liver_extent = LIVER_Z_EXTENT_PRIOR_MM
        consistent = []
        spread = None
        extent_source = "liver_prior"

    rows = []
    for phase, item in per_phase.items():
        record = item["record"]
        own = item["vertebra"]["spacing_mm"]
        own_pairs = item["vertebra"]["n_pairs"] if item["vertebra"]["kind"] == "level_specific" else 0
        if item["liver_slices"]:
            spacing = liver_extent / item["liver_slices"]
            basis = f"{extent_source}_shared_liver_extent"
            if own and own_pairs >= 2 and abs(math.log(own / spacing)) > math.log(1.0 + PHASE_EXTENT_INCONSISTENCY):
                # This phase's liver mask does not span the same range as the
                # other phases (truncated scan or mask); its own vertebrae win.
                spacing = own
                basis = "vertebrae_this_phase_liver_extent_inconsistent"
        elif own:
            spacing = own
            basis = "vertebrae_this_phase_only"
        else:
            spacing = LIVER_Z_EXTENT_PRIOR_MM / max(1, int(round(record["size"][2] * 0.96)))
            basis = "volume_slice_count_prior"
        if basis.startswith("vertebrae_this_phase"):
            confidence = "medium" if own_pairs >= 2 else "low"
        elif extent_source == "vertebrae" and len(consistent) >= 2 and spread is not None and spread <= 0.10:
            confidence = "high"
        elif extent_source == "vertebrae" and spread is not None and spread <= 0.20:
            confidence = "medium"
        else:
            confidence = "low"
        orientation = record.get("orientation", {})
        rows.append(
            {
                "patient_id": record["patient_id"],
                "phase": phase,
                "plc_phase": next(k for k, v in PLC_PHASES.items() if v == phase),
                "size_x": record["size"][0],
                "size_y": record["size"][1],
                "size_z": record["size"][2],
                "spacing_x_mm": PLC_IN_PLANE_SPACING_MM,
                "spacing_y_mm": PLC_IN_PLANE_SPACING_MM,
                "spacing_z_mm": round(spacing, 4),
                "spacing_z_continuous_mm": round(spacing, 4),
                "spacing_z_snap": "continuous",
                "spacing_z_source": basis,
                "spacing_z_confidence": confidence,
                "k_axis_flip": int(bool(record.get("k_axis_flip"))),
                "orientation_source": orientation.get("source", ""),
                "orientation_evidence": orientation.get("patient_evidence", ""),
                "vertebra_order_score": record.get("vertebra_order_score", ""),
                "liver_slices": item["liver_slices"],
                "liver_slices_source": item["liver_source"],
                "liver_z_extent_mm": round(liver_extent, 1),
                "liver_z_extent_source": extent_source,
                "liver_z_extent_phase_spread": round(spread, 3) if spread is not None else "",
                "vertebra_spacing_this_phase_mm": round(own, 4) if own else "",
                "vertebra_pairs": item["vertebra"]["n_pairs"],
                "vertebra_pair_kind": item["vertebra"]["kind"] or "",
                "vertebra_levels": "+".join(v["level"] for v in record.get("vertebrae", [])),
                "initial_guess_mm": round(record["slice_spacing_guess_mm"], 4),
                "totalsegmentator_liver_native_file": relative_fallback_path(
                    record.get("totalsegmentator_liver_native_file"), output_root
                ),
                "geometry_version": GEOMETRY_VERSION,
            }
        )
    rows.sort(key=lambda r: PHASES.index(r["phase"]))
    return rows


def run_solve(args: argparse.Namespace) -> None:
    cases = select_cases(args)
    done = load_measurements(measurements_path(args.output_root))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        records = {phase: done[(case["patient_id"], phase)] for phase in PHASES if (case["patient_id"], phase) in done}
        if len(records) != len(PHASES):
            missing.append(case["patient_id"])
            continue
        rows.extend(solve_patient(records, args.reference_scale, args.output_root))
    if missing:
        raise SystemExit(
            f"{len(missing)} PLC patients lack measurements for all four phases "
            f"(first: {missing[:5]}). Run 'measure' first."
        )
    path = geometry_csv_path(args.output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)

    spacings = [r["spacing_z_mm"] for r in rows]
    by_conf: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_orient: dict[str, int] = defaultdict(int)
    for r in rows:
        by_conf[r["spacing_z_confidence"]] += 1
        by_source[r["spacing_z_source"]] += 1
        by_orient[r["orientation_source"]] += 1
    summary = {
        "geometry_version": GEOMETRY_VERSION,
        "patients": len(rows) // len(PHASES),
        "volumes": len(rows),
        "reference_scale": args.reference_scale,
        "spacing_z_mm_percentiles": {
            f"p{p}": round(float(np.percentile(spacings, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)
        },
        "k_axis_flipped_volumes": sum(r["k_axis_flip"] for r in rows),
        "orientation_source_counts": dict(sorted(by_orient.items())),
        "confidence_counts": dict(sorted(by_conf.items())),
        "source_counts": dict(sorted(by_source.items())),
        "liver_extent_mm_percentiles": {
            f"p{p}": round(float(np.percentile([r["liver_z_extent_mm"] for r in rows], p)), 1)
            for p in (5, 50, 95)
        },
    }
    (geometry_dir(args.output_root) / "plc_geometry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "download-weights":
        download_weights()
        return
    if args.command in ("measure", "all"):
        run_measure(args)
    if args.command in ("solve", "all"):
        run_solve(args)


if __name__ == "__main__":
    main()
