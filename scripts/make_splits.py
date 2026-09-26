"""Reproducible patient-level train / validation / test split of a unified_v2 root.

Two protocols:

small-held-out (default)
    Every patient with at least one small lesion (any lesion in the 0-5, 5-10
    or 10-15 mm bins by default, --small-bins) is kept out of training: these
    patients form the test subset (a fraction can go to validation with
    --small-val-fraction).  Patients whose lesions are all > 15 mm, and
    lesion-negative patients, are split train / val / test by --ratios
    (default 0.80 / 0.10 / 0.10) with stratification over dataset, diagnosis
    and lesion presence, so the test subset also measures large-lesion
    performance.  This is the protocol for the question "does a model that
    only ever saw large lesions learn to find small ones": the training set
    contains no lesion <= 15 mm.

stratified
    Every stratum (dataset, coarse diagnosis, lesion-negative, presence of
    lesions in each size bin, dataset x bin) is represented in every subset in
    proportion to --ratios (default 0.70 / 0.10 / 0.20), so size-binned metrics
    can be computed on validation and test while the model also trains on
    small lesions.

Algorithm: iterative stratification (Sechidis, Tsoumakas, Vlahavas 2011).
Cases carry a set of labels; the label with the fewest unassigned cases is
handled first; each of its cases goes to the subset that still needs that
label most (ties: the subset that needs cases most, then a seeded random
order).  Everything is deterministic for a given manifest and seed.

Evaluation subsets are "clean" by default: a case whose tumour mask was
propagated through a failed registration (tumour_mask_reliable = 0) or that
has a phase below the registration gate is kept out of validation and test
and forced into train, where phase dropout handles it.  Under small-held-out a
flagged patient with a small lesion cannot go to train either, so it is
excluded from all subsets.  Disable with --no-clean-eval.

Outputs (in --out, default <root>/splits/):
  <name>.json          subsets, per-case labels, metadata, stratum counts
  <name>_cases.csv     one row per case with its subset and stratification labels
  <name>_summary.csv   subset x dataset counts of cases and lesions per size bin
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SUBSETS = ("train", "val", "test")
SIZE_BINS = ("0_5mm", "5_10mm", "10_15mm", "gt_15mm")
BIN_LABELS = {"0_5mm": "0-5 mm", "5_10mm": "5-10 mm", "10_15mm": "10-15 mm", "gt_15mm": "> 15 mm"}


def case_labels(row: dict[str, str]) -> list[str]:
    labels = [f"dataset={row['dataset']}", f"diagnosis={row['diagnosis_coarse']}"]
    total = int(row["lesion_count_total"] or 0)
    labels.append("lesions=present" if total > 0 else "lesions=none")
    for size_bin in SIZE_BINS:
        if int(row[f"lesion_count_{size_bin}"] or 0) > 0:
            labels.append(f"has_{size_bin}")
            # joint label: small lesions are rare outside MCT-LTDiag, so the
            # dataset x bin combination must be balanced explicitly
            labels.append(f"{row['dataset']}:has_{size_bin}")
    return labels


def is_clean(row: dict[str, str]) -> tuple[bool, list[str]]:
    reasons = []
    if row["tumour_mask_reliable"] != "1":
        reasons.append("tumour_mask_unreliable")
    if row["usable_phases"] != "NC+AP+PVP+DP":
        reasons.append(f"usable_phases={row['usable_phases']}")
    return not reasons, reasons


def iterative_stratification(
    cases: dict[str, list[str]], ratios: dict[str, float], rng: random.Random
) -> dict[str, str]:
    """Assign every case id to a subset; deterministic for a given rng state."""
    remaining = set(cases)
    desired_cases = {s: ratios[s] * len(cases) for s in SUBSETS}
    label_total: Counter[str] = Counter(label for labels in cases.values() for label in labels)
    desired_label = {s: {label: ratios[s] * n for label, n in label_total.items()} for s in SUBSETS}
    assignment: dict[str, str] = {}
    order = {case_id: rng.random() for case_id in cases}  # one seeded tie-break value per case

    while remaining:
        counts = Counter(label for case_id in remaining for label in cases[case_id])
        if not counts:
            label = None
        else:
            # rarest label first; deterministic tie-break on the label name
            label = min(counts, key=lambda l: (counts[l], l))
        pool = [c for c in remaining if label is None or label in cases[c]]
        pool.sort(key=lambda c: order[c])
        for case_id in pool:
            if label is None:
                best = max(SUBSETS, key=lambda s: (desired_cases[s], s))
            else:
                best = max(
                    SUBSETS,
                    key=lambda s: (round(desired_label[s][label], 9), round(desired_cases[s], 9), order[case_id] if s == "train" else 0),
                )
            assignment[case_id] = best
            remaining.discard(case_id)
            desired_cases[best] -= 1
            for l in cases[case_id]:
                desired_label[best][l] -= 1
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="unified_v2 output root (dataset_manifest.csv inside).")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default <root>/splits).")
    parser.add_argument("--protocol", choices=("small-held-out", "stratified"), default="small-held-out")
    parser.add_argument("--name", default=None, help="Base name of the output files (default unified_v2_<protocol>_seed<seed>).")
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"),
        help="Subset shares; default 0.80/0.10/0.10 for small-held-out (large-only and negative patients), 0.70/0.10/0.20 for stratified.",
    )
    parser.add_argument("--small-bins", nargs="+", choices=SIZE_BINS, default=["0_5mm", "5_10mm", "10_15mm"],
                        help="Size bins whose presence makes a patient a small-lesion patient (small-held-out).")
    parser.add_argument("--small-val-fraction", type=float, default=0.0,
                        help="Share of small-lesion patients placed in validation instead of test (small-held-out).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict to these datasets (default all).")
    parser.add_argument("--no-clean-eval", action="store_true", help="Allow flagged cases in validation and test.")
    args = parser.parse_args()
    if args.ratios is None:
        args.ratios = (0.80, 0.10, 0.10) if args.protocol == "small-held-out" else (0.70, 0.10, 0.20)
    if args.name is None:
        args.name = f"unified_v2_{args.protocol.replace('-', '_')}_seed{args.seed}"
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("--ratios must sum to 1")
    if not 0.0 <= args.small_val_fraction <= 1.0:
        raise SystemExit("--small-val-fraction must be within [0, 1]")
    ratios = dict(zip(SUBSETS, args.ratios))
    out = args.out or (args.root / "splits")
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.root / "dataset_manifest.csv"
    rows = [r for r in csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")) if r["status"] == "complete"]
    if args.datasets:
        rows = [r for r in rows if r["dataset"] in set(args.datasets)]
    rows.sort(key=lambda r: r["case_id"])
    if not rows or "lesion_count_total" not in rows[0]:
        raise SystemExit("dataset_manifest.csv has no lesion columns; run prepare_hierarchical_labels.py first")

    rng = random.Random(args.seed)
    forced_train: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    small_ids: set[str] = set()
    pool: dict[str, list[str]] = {}
    small_pool: dict[str, list[str]] = {}

    def is_small(row: dict[str, str]) -> bool:
        return any(int(row[f"lesion_count_{b}"] or 0) > 0 for b in args.small_bins)

    for row in rows:
        clean, reasons = is_clean(row)
        small = args.protocol == "small-held-out" and is_small(row)
        if small:
            small_ids.add(row["case_id"])
        if not clean and not args.no_clean_eval:
            if small:
                excluded[row["case_id"]] = reasons  # cannot train on it, cannot evaluate on it
            else:
                forced_train[row["case_id"]] = reasons
        elif small:
            small_pool[row["case_id"]] = case_labels(row)
        else:
            pool[row["case_id"]] = case_labels(row)
    n_total = len(rows) - len(excluded)
    assignment: dict[str, str] = {}
    if args.protocol == "small-held-out":
        # Small-lesion patients: test, or val for the requested fraction.
        if small_pool:
            small_ratios = {"train": 0.0, "val": args.small_val_fraction, "test": 1.0 - args.small_val_fraction}
            assignment.update(iterative_stratification(small_pool, small_ratios, rng))
        # Large-only and negative patients by --ratios; forced-train cases take
        # part of the train share.
        pool_ratios = {
            "train": max(0.0, ratios["train"] * len(pool) - len(forced_train)),
            "val": ratios["val"] * len(pool),
            "test": ratios["test"] * len(pool),
        }
    else:
        pool_ratios = {
            "train": max(0.0, ratios["train"] * n_total - len(forced_train)),
            "val": ratios["val"] * n_total,
            "test": ratios["test"] * n_total,
        }
    total = sum(pool_ratios.values())
    pool_ratios = {k: v / total for k, v in pool_ratios.items()}
    assignment.update(iterative_stratification(pool, pool_ratios, rng))
    for case_id in forced_train:
        assignment[case_id] = "train"
    rows = [row for row in rows if row["case_id"] not in excluded]

    by_case = {row["case_id"]: row for row in rows}
    subsets = {s: sorted(c for c, a in assignment.items() if a == s) for s in SUBSETS}

    # Summary per subset x dataset
    summary_rows = []
    for subset in SUBSETS:
        for dataset in sorted({r["dataset"] for r in rows}):
            members = [by_case[c] for c in subsets[subset] if by_case[c]["dataset"] == dataset]
            entry = {
                "subset": subset,
                "dataset": dataset,
                "cases": len(members),
                "lesion_positive_cases": sum(int(m["lesion_count_total"] or 0) > 0 for m in members),
                "lesions_total": sum(int(m["lesion_count_total"] or 0) for m in members),
            }
            for size_bin in SIZE_BINS:
                entry[f"lesions_{size_bin}"] = sum(int(m[f"lesion_count_{size_bin}"] or 0) for m in members)
                entry[f"cases_with_{size_bin}"] = sum(int(m[f"lesion_count_{size_bin}"] or 0) > 0 for m in members)
            for diagnosis in ("control", "HH", "HCC", "ICC", "cHCC-CCA", "metastasis"):
                entry[f"diagnosis_{diagnosis}"] = sum(m["diagnosis_coarse"] == diagnosis for m in members)
            entry["forced_train_flagged"] = sum(m["case_id"] in forced_train for m in members)
            entry["small_lesion_patients"] = sum(m["case_id"] in small_ids for m in members)
            summary_rows.append(entry)
    with (out / f"{args.name}_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    case_rows = []
    for row in rows:
        case_id = row["case_id"]
        case_rows.append(
            {
                "case_id": case_id,
                "dataset": row["dataset"],
                "patient_id": row["patient_id"],
                "subset": assignment[case_id],
                "diagnosis_coarse": row["diagnosis_coarse"],
                "diagnosis_fine": row["diagnosis_fine"],
                "lesion_count_total": row["lesion_count_total"],
                **{f"lesion_count_{b}": row[f"lesion_count_{b}"] for b in SIZE_BINS},
                "usable_phases": row["usable_phases"],
                "tumour_mask_reliable": row["tumour_mask_reliable"],
                "sanity_pass": row["sanity_pass"],
                "native_spacing_z_pvp_mm": row["native_spacing_z_pvp_mm"],
                "spacing_confidence": row["spacing_confidence"],
                "small_lesion_patient": int(case_id in small_ids),
                "forced_train_reason": ";".join(forced_train.get(case_id, [])),
                "stratification_labels": ";".join(pool.get(case_id, small_pool.get(case_id, []))),
            }
        )
    with (out / f"{args.name}_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    all_labels = {**pool, **small_pool}
    label_counts = {
        s: dict(sorted(Counter(l for c in subsets[s] for l in all_labels.get(c, [])).items())) for s in SUBSETS
    }
    payload = {
        "name": args.name,
        "protocol": args.protocol,
        "protocol_rule": (
            f"patients with any lesion in {args.small_bins} never train; {1 - args.small_val_fraction:.2f} of them test, "
            f"{args.small_val_fraction:.2f} val; other patients split {ratios} with stratification"
            if args.protocol == "small-held-out"
            else f"all patients split {ratios} with stratification over every label"
        ),
        "small_bins": args.small_bins if args.protocol == "small-held-out" else [],
        "small_lesion_patients": {s: sum(c in small_ids for c in subsets[s]) for s in SUBSETS},
        "created": "deterministic: same manifest + same seed + same options -> same split",
        "source_manifest": str(manifest_path),
        "manifest_cases": len(rows) + len(excluded),
        "seed": args.seed,
        "ratios": ratios,
        "clean_eval": not args.no_clean_eval,
        "clean_eval_rule": "val/test require tumour_mask_reliable = 1 and all four phases usable; other large-only patients are forced into train; flagged small-lesion patients are excluded",
        "forced_train": forced_train,
        "excluded": excluded,
        "size_bins_mm": {b: BIN_LABELS[b] for b in SIZE_BINS},
        "size_definition": "equivalent spherical diameter of a 26-connected component on the 1 mm grid (lesion_size_statistics.csv)",
        "stratification_labels": "dataset, diagnosis_coarse, lesions present/none, has_<bin> and dataset:has_<bin> for each size bin",
        "algorithm": "iterative stratification (Sechidis et al. 2011), rarest label first, deterministic tie-breaks",
        "counts": {s: len(subsets[s]) for s in SUBSETS},
        "label_counts": label_counts,
        "summary": summary_rows,
        "subsets": subsets,
    }
    (out / f"{args.name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"protocol {args.protocol}; cases {len(rows)}: " + ", ".join(f"{s} {len(subsets[s])}" for s in SUBSETS)
        + f"; flagged forced into train: {len(forced_train)}; excluded: {len(excluded)}"
        + (
            f"; small-lesion patients val/test: {payload['small_lesion_patients']['val']}/{payload['small_lesion_patients']['test']}"
            if args.protocol == "small-held-out"
            else ""
        )
    )
    header = f"{'subset':6s} {'dataset':11s} {'cases':>5s} {'pos':>4s} " + " ".join(f"{BIN_LABELS[b]:>9s}" for b in SIZE_BINS)
    print(header)
    for entry in summary_rows:
        print(f"{entry['subset']:6s} {entry['dataset']:11s} {entry['cases']:5d} {entry['lesion_positive_cases']:4d} " + " ".join(f"{entry[f'lesions_{b}']:4d}/{entry[f'cases_with_{b}']:<4d}" for b in SIZE_BINS))
    print("(lesions/cases per size bin)")


if __name__ == "__main__":
    main()
