"""Reproducible patient-level train / validation / test split of a unified_v2 root.

The split is stratified so that every evaluation stratum is represented in
every subset in proportion: dataset, coarse diagnosis, lesion-negative cases
and, most importantly, the presence of lesions in each size bin
(0-5, 5-10, 10-15, > 15 mm equivalent spherical diameter) so that size-binned
metrics can be computed on validation and test.  Small lesions are rare
outside MCT-LTDiag, which is why plain random splitting is not enough.

Algorithm: iterative stratification (Sechidis, Tsoumakas, Vlahavas 2011).
Cases carry a set of labels; the label with the fewest unassigned cases is
handled first; each of its cases goes to the subset that still needs that
label most (ties: the subset that needs cases most, then a seeded random
order).  Everything is deterministic for a given manifest and seed.

Evaluation subsets are "clean" by default: a case whose tumour mask was
propagated through a failed registration (tumour_mask_reliable = 0) or that
has a phase below the registration gate is kept out of validation and test
and forced into train, where phase dropout handles it.  Disable with
--no-clean-eval.

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
    parser.add_argument("--name", default="unified_v2_split_seed0", help="Base name of the output files.")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.10, 0.20), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict to these datasets (default all).")
    parser.add_argument("--no-clean-eval", action="store_true", help="Allow flagged cases in validation and test.")
    args = parser.parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("--ratios must sum to 1")
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
    pool: dict[str, list[str]] = {}
    for row in rows:
        clean, reasons = is_clean(row)
        if not clean and not args.no_clean_eval:
            forced_train[row["case_id"]] = reasons
        else:
            pool[row["case_id"]] = case_labels(row)
    # The forced-train cases consume part of the train share so that the
    # stratified pool still produces the requested overall proportions.
    n_total = len(rows)
    pool_ratios = {
        "train": max(0.0, ratios["train"] * n_total - len(forced_train)) / len(pool),
        "val": ratios["val"] * n_total / len(pool),
        "test": ratios["test"] * n_total / len(pool),
    }
    total = sum(pool_ratios.values())
    pool_ratios = {k: v / total for k, v in pool_ratios.items()}
    assignment = iterative_stratification(pool, pool_ratios, rng)
    for case_id in forced_train:
        assignment[case_id] = "train"

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
                "forced_train_reason": ";".join(forced_train.get(case_id, [])),
                "stratification_labels": ";".join(pool.get(case_id, [])),
            }
        )
    with (out / f"{args.name}_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    label_counts = {
        s: dict(sorted(Counter(l for c in subsets[s] for l in pool.get(c, [])).items())) for s in SUBSETS
    }
    payload = {
        "name": args.name,
        "created": "deterministic: same manifest + same seed + same options -> same split",
        "source_manifest": str(manifest_path),
        "manifest_cases": n_total,
        "seed": args.seed,
        "ratios": ratios,
        "clean_eval": not args.no_clean_eval,
        "clean_eval_rule": "val/test require tumour_mask_reliable = 1 and all four phases usable; others are forced into train",
        "forced_train": forced_train,
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

    print(f"cases {n_total}: " + ", ".join(f"{s} {len(subsets[s])}" for s in SUBSETS) + f"; flagged cases forced into train: {len(forced_train)}")
    header = f"{'subset':6s} {'dataset':11s} {'cases':>5s} {'pos':>4s} " + " ".join(f"{BIN_LABELS[b]:>9s}" for b in SIZE_BINS)
    print(header)
    for entry in summary_rows:
        print(f"{entry['subset']:6s} {entry['dataset']:11s} {entry['cases']:5d} {entry['lesion_positive_cases']:4d} " + " ".join(f"{entry[f'lesions_{b}']:4d}/{entry[f'cases_with_{b}']:<4d}" for b in SIZE_BINS))
    print("(lesions/cases per size bin)")


if __name__ == "__main__":
    main()
