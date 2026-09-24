"""Read-only completion check of a unified_v2 output root.

For every case directory: the four image channels and both masks exist, share
size / spacing / origin / direction, are 1 x 1 x 1 mm, images are int16 and
masks are binary uint8, and case.json carries the expected pipeline version.
Prints per-dataset counts of complete cases, usable phases, sanity failures
and the failure inventory.  Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPENDENCIES = PROJECT_ROOT / "tmp" / "python_deps"
if LOCAL_DEPENDENCIES.exists():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))

import numpy as np
import SimpleITK as sitk

FILES = (
    "image_nc.nii.gz",
    "image_ap.nii.gz",
    "image_pvp.nii.gz",
    "image_dp.nii.gz",
    "liver_mask.nii.gz",
    "tumor_mask.nii.gz",
)
MASKS = {"liver_mask.nii.gz", "tumor_mask.nii.gz"}


def check_case(args: tuple[Path, str]) -> dict[str, object]:
    case_dir, expected_version = args
    problems: list[str] = []
    record: dict[str, object] = {}
    try:
        record = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"case.json unreadable: {exc}")
    if record.get("pipeline_version") != expected_version:
        problems.append(f"pipeline_version {record.get('pipeline_version')!r}")
    reference = None
    for name in FILES:
        path = case_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{name} missing")
            continue
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        size, spacing, origin, direction = reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection()
        pixel = sitk.GetPixelIDValueAsString(reader.GetPixelID())
        if reference is None:
            reference = (size, spacing, origin, direction)
        elif not (
            size == reference[0]
            and np.allclose(spacing, reference[1], atol=1e-6)
            and np.allclose(origin, reference[2], atol=1e-4)
            and np.allclose(direction, reference[3], atol=1e-6)
        ):
            problems.append(f"{name} grid differs from image_nc")
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-6):
            problems.append(f"{name} spacing {spacing}")
        if name in MASKS:
            if pixel != "8-bit unsigned integer":
                problems.append(f"{name} dtype {pixel}")
            else:
                mask_image = sitk.ReadImage(str(path))
                values = np.unique(sitk.GetArrayFromImage(mask_image))
                if not set(values.tolist()) <= {0, 1}:
                    problems.append(f"{name} not binary")
        elif pixel != "16-bit signed integer":
            problems.append(f"{name} dtype {pixel}")
    return {
        "case_id": case_dir.name,
        "dataset": record.get("dataset", case_dir.name.split("_")[0]),
        "problems": problems,
        "usable_phases": record.get("usable_phases", []),
        "sanity_pass": bool(record.get("sanity", {}).get("pass", False)),
        "sanity_flags": record.get("sanity", {}).get("flags", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pipeline output root (contains cases/ and summary.json).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    expected_version = summary["pipeline_version"]
    case_dirs = sorted(path for path in (args.root / "cases").iterdir() if path.is_dir())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(check_case, [(path, expected_version) for path in case_dirs], chunksize=8))

    complete: Counter[str] = Counter()
    broken: list[dict[str, object]] = []
    usable: dict[str, Counter[str]] = defaultdict(Counter)
    sanity_fail: Counter[str] = Counter()
    flag_kinds: Counter[str] = Counter()
    for result in results:
        dataset = str(result["dataset"])
        if result["problems"]:
            broken.append(result)
            continue
        complete[dataset] += 1
        usable[dataset]["+".join(result["usable_phases"])] += 1
        if not result["sanity_pass"]:
            sanity_fail[dataset] += 1
            for flag in result["sanity_flags"]:
                flag_kinds[flag.split("_", 1)[1] if flag[:2] in {"NC", "AP", "DP"} or flag[:3] == "PVP" else flag] += 1
    print(f"pipeline_version: {expected_version}")
    print(f"case directories: {len(case_dirs)}; complete and consistent: {sum(complete.values())}; broken: {len(broken)}")
    for dataset in sorted(complete):
        print(f"  {dataset}: {complete[dataset]} complete; usable-phase sets: {dict(usable[dataset])}; sanity failures: {sanity_fail[dataset]}")
    if flag_kinds:
        print("sanity flag kinds (phase prefix stripped):")
        for flag, count in flag_kinds.most_common(20):
            print(f"  {count:4d}  {flag}")
    for result in broken[:20]:
        print(f"BROKEN {result['case_id']}: {'; '.join(result['problems'])}")
    failures = json.loads((args.root / "failures.json").read_text(encoding="utf-8"))
    print(f"failures.json entries: {len(failures)}")
    return 1 if broken or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
