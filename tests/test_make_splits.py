from __future__ import annotations

import random
import unittest
from collections import Counter

from scripts.make_splits import SUBSETS, case_labels, is_clean, iterative_stratification


def synthetic_cases(n: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    cases = {}
    for i in range(n):
        dataset = rng.choice(["A", "B", "C"])
        labels = [f"dataset={dataset}", "lesions=present" if rng.random() < 0.9 else "lesions=none"]
        if rng.random() < 0.2:
            labels += ["has_0_5mm", f"{dataset}:has_0_5mm"]
        cases[f"case{i:04d}"] = labels
    return cases


class SplitTests(unittest.TestCase):
    def test_same_input_and_seed_give_the_same_split(self) -> None:
        cases = synthetic_cases(300, 1)
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}
        a = iterative_stratification(cases, ratios, random.Random(0))
        b = iterative_stratification(cases, ratios, random.Random(0))
        c = iterative_stratification(cases, ratios, random.Random(1))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(set(a), set(cases))

    def test_rare_labels_are_spread_in_proportion(self) -> None:
        cases = synthetic_cases(600, 2)
        ratios = {"train": 0.7, "val": 0.1, "test": 0.2}
        assignment = iterative_stratification(cases, ratios, random.Random(0))
        counts = Counter(assignment.values())
        for subset in SUBSETS:
            self.assertAlmostEqual(counts[subset] / 600, ratios[subset], delta=0.02)
        rare = [c for c, labels in cases.items() if "has_0_5mm" in labels]
        rare_counts = Counter(assignment[c] for c in rare)
        for subset in SUBSETS:
            self.assertAlmostEqual(rare_counts[subset] / len(rare), ratios[subset], delta=0.06)

    def test_labels_and_cleanliness_from_manifest_row(self) -> None:
        row = {"dataset": "PLC-CECT", "diagnosis_coarse": "HCC", "lesion_count_total": "3", "lesion_count_0_5mm": "0",
               "lesion_count_5_10mm": "1", "lesion_count_10_15mm": "0", "lesion_count_gt_15mm": "2",
               "tumour_mask_reliable": "1", "usable_phases": "NC+AP+PVP+DP"}
        self.assertEqual(case_labels(row), ["dataset=PLC-CECT", "diagnosis=HCC", "lesions=present", "has_5_10mm", "PLC-CECT:has_5_10mm", "has_gt_15mm", "PLC-CECT:has_gt_15mm"])
        self.assertEqual(is_clean(row), (True, []))
        row["usable_phases"] = "AP+PVP+DP"
        self.assertEqual(is_clean(row)[0], False)


if __name__ == "__main__":
    unittest.main()
