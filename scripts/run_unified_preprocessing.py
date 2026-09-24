"""Run the complete three-dataset preprocessing pipeline (unified_v2).

The source releases are read only. The ordered stages are:
0. PLC-CECT geometry recovery (slice order, in-plane and slice spacing) with
   the TotalSegmentator 3 mm model, written to manifests/plc_geometry.csv;
1. four-phase conversion: HU recovery, liver-mask driven registration with a
   Dice gate, liver-region crop and one resampling to a 1 mm isotropic grid;
2. hierarchical patient labels and lesion-size statistics.

Any subset of the three datasets can be processed (``--datasets``), and each
dataset's release folder can be given explicitly (``--mct-root``,
``--plc-root``, ``--waw-root``) or found under ``--raw-root/<dataset name>``.
Stage 0 is resumable and skips volumes that were already measured.  Stage 1
skips cases whose case.json already carries the current pipeline version, so
an interrupted run can simply be restarted.  Run this script with the
interpreter of the preprocessing environment: stage 0 needs TotalSegmentator
and a GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAW_PHASES = {"0", "1", "2", "3"}
WAW_MASK_PATTERN = re.compile(r"/(\d+)_(\d+)_(\d+)_tumor_seg\.nrrd$")
ALL_DATASETS = ("PLC-CECT", "MCT-LTDiag", "WAW-TACE")
DATASET_FOLDERS = {"MCT-LTDiag": "MCT-LTDiag", "PLC-CECT": "PLC-CECT", "WAW-TACE": "WAW-TACE"}


def patient_sort_key(patient_id: str) -> tuple[int, int | str]:
    if patient_id.isdigit():
        return (0, int(patient_id))
    return (1, patient_id)


def discover_waw_four_phase_patients(waw_root: Path) -> list[str]:
    """Return WAW patients with four CT phases and one annotated phase."""
    metadata_path = waw_root / "ct_hcc_metadata.csv"
    mask_archive = waw_root / "tumor_masks.zip"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing WAW metadata: {metadata_path}")
    if not mask_archive.is_file():
        raise FileNotFoundError(f"Missing WAW tumour masks: {mask_archive}")

    phases_by_patient: dict[str, set[str]] = defaultdict(set)
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            phases_by_patient[row["patient_id"]].add(row["ct_phase"])

    annotated_phases: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(mask_archive) as archive:
        for info in archive.infolist():
            match = WAW_MASK_PATTERN.search(info.filename)
            if match:
                annotated_phases[match.group(1)].add(match.group(2))

    return sorted(
        (
            patient_id
            for patient_id, phases in phases_by_patient.items()
            if phases == WAW_PHASES and len(annotated_phases.get(patient_id, set())) == 1
        ),
        key=patient_sort_key,
    )


def write_waw_selection(path: Path, patient_ids: Iterable[str]) -> None:
    patient_ids = list(patient_ids)
    if not patient_ids:
        raise ValueError("WAW-TACE selection cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        delete=False,
        dir=path.parent,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id"])
        writer.writeheader()
        writer.writerows({"patient_id": patient_id} for patient_id in patient_ids)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def resolve_dataset_roots(
    datasets: Sequence[str],
    raw_root: Path | None,
    explicit: dict[str, Path | None],
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for dataset in datasets:
        root = explicit.get(dataset)
        if root is None and raw_root is not None:
            root = raw_root / DATASET_FOLDERS[dataset]
        if root is None:
            raise SystemExit(f"{dataset}: give --raw-root or the dataset's own --*-root option")
        if not root.is_dir():
            raise SystemExit(f"{dataset}: release folder not found: {root}")
        roots[dataset] = root
    return roots


def build_stage_commands(
    *,
    python_executable: Path,
    project_root: Path,
    dataset_roots: dict[str, Path],
    output_root: Path,
    waw_selection: Path,
    workers: int,
    datasets: Sequence[str] = ALL_DATASETS,
    replace_datasets: Sequence[str] = (),
    threads_per_worker: int = 2,
    reference_scale: float = 1.0,
    include_geometry_stage: bool = True,
) -> list[list[str]]:
    python = str(python_executable)
    scripts = project_root / "scripts"
    commands: list[list[str]] = []
    if "PLC-CECT" in datasets and include_geometry_stage:
        commands.append(
            [
                python,
                str(scripts / "estimate_plc_geometry.py"),
                "all",
                "--plc-root",
                str(dataset_roots["PLC-CECT"]),
                "--output-root",
                str(output_root),
                "--reference-scale",
                str(reference_scale),
            ]
        )
    conversion = [
        python,
        str(scripts / "preprocess_multiphase_ct.py"),
        "--output-root",
        str(output_root),
        "--waw-selection",
        str(waw_selection),
        "--plc-geometry",
        str(output_root / "manifests" / "plc_geometry.csv"),
        "--workers",
        str(workers),
        "--threads-per-worker",
        str(threads_per_worker),
        "--datasets",
        *datasets,
    ]
    for dataset, option in (("MCT-LTDiag", "--mct-root"), ("PLC-CECT", "--plc-root"), ("WAW-TACE", "--waw-root")):
        if dataset in dataset_roots:
            conversion.extend([option, str(dataset_roots[dataset])])
    if replace_datasets:
        conversion.extend(["--replace-datasets", *replace_datasets])
    commands.append(conversion)
    commands.append(
        [
            python,
            str(scripts / "prepare_hierarchical_labels.py"),
            "--root",
            str(output_root),
        ]
    )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Folder that holds the release folders MCT-LTDiag/, PLC-CECT/ and WAW-TACE/.",
    )
    parser.add_argument("--mct-root", type=Path, default=None, help="MCT-LTDiag release folder (overrides --raw-root).")
    parser.add_argument("--plc-root", type=Path, default=None, help="PLC-CECT release folder (overrides --raw-root).")
    parser.add_argument("--waw-root", type=Path, default=None, help="WAW-TACE release folder (overrides --raw-root).")
    parser.add_argument("--output-root", type=Path, required=True, help="Where cases/ and manifests/ are written.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=list(ALL_DATASETS),
        help="Datasets to process; any subset of the three.",
    )
    parser.add_argument(
        "--replace-datasets",
        nargs="*",
        choices=ALL_DATASETS,
        default=(),
        help="Rebuild and atomically replace cases of these datasets that are not at the current pipeline version.",
    )
    parser.add_argument(
        "--reference-scale",
        type=float,
        default=1.0,
        help="Population calibration factor for the PLC vertebra reference distances (stage 0).",
    )
    parser.add_argument(
        "--reuse-plc-geometry",
        action="store_true",
        help=(
            "Skip stage 0 when <output-root>/manifests/plc_geometry.csv already exists "
            "(for example copied from reference/), instead of re-estimating the PLC geometry."
        ),
    )
    parser.add_argument(
        "--expected-waw-cases",
        type=int,
        default=164,
        help="Fail before processing if automatic WAW selection finds another count; use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect sources and print the ordered commands without writing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.expected_waw_cases < 0:
        raise SystemExit("--expected-waw-cases cannot be negative")
    unselected = set(args.replace_datasets) - set(args.datasets)
    if unselected:
        raise SystemExit("--replace-datasets must be a subset of --datasets: " + ", ".join(sorted(unselected)))
    dataset_roots = resolve_dataset_roots(
        args.datasets,
        args.raw_root,
        {"MCT-LTDiag": args.mct_root, "PLC-CECT": args.plc_root, "WAW-TACE": args.waw_root},
    )

    selection_path = args.output_root / "manifests" / "waw_four_phase_patients.csv"
    waw_patients: list[str] = []
    if "WAW-TACE" in args.datasets:
        waw_patients = discover_waw_four_phase_patients(dataset_roots["WAW-TACE"])
        if args.expected_waw_cases and len(waw_patients) != args.expected_waw_cases:
            raise SystemExit(
                f"Expected {args.expected_waw_cases} WAW-TACE cases, found {len(waw_patients)}"
            )
    geometry_csv = args.output_root / "manifests" / "plc_geometry.csv"
    reuse_geometry = bool(args.reuse_plc_geometry and geometry_csv.is_file())
    if args.reuse_plc_geometry and not geometry_csv.is_file():
        print(f"--reuse-plc-geometry: {geometry_csv} not found, stage 0 will run", flush=True)
    commands = build_stage_commands(
        python_executable=Path(sys.executable),
        project_root=PROJECT_ROOT,
        dataset_roots=dataset_roots,
        output_root=args.output_root,
        waw_selection=selection_path,
        workers=args.workers,
        datasets=args.datasets,
        replace_datasets=args.replace_datasets,
        threads_per_worker=args.threads_per_worker,
        reference_scale=args.reference_scale,
        include_geometry_stage=not reuse_geometry,
    )
    if reuse_geometry:
        print(f"Reusing existing PLC geometry: {geometry_csv}", flush=True)

    for dataset, root in dataset_roots.items():
        print(f"{dataset}: {root}", flush=True)
    if waw_patients:
        print(f"WAW-TACE four-phase cases: {len(waw_patients)}", flush=True)
    if args.dry_run:
        if waw_patients:
            print(f"Selection manifest would be written to: {selection_path}")
        for index, command in enumerate(commands, start=1):
            print(f"Stage {index}: {subprocess.list2cmdline(command)}")
        return 0

    if waw_patients:
        write_waw_selection(selection_path, waw_patients)
    for index, command in enumerate(commands, start=1):
        print(f"Starting stage {index}/{len(commands)}: {Path(command[1]).name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"Unified preprocessing complete: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
