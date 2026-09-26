from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.evaluate_by_lesion_size import aggregate, esd_mm, evaluate_case, size_bin, wilson


class SizeBinnedEvaluationTests(unittest.TestCase):
    def test_bins_match_the_pipeline_definition(self) -> None:
        self.assertEqual(size_bin(5.0), "0_5mm")
        self.assertEqual(size_bin(5.01), "5_10mm")
        self.assertEqual(size_bin(15.0), "10_15mm")
        self.assertEqual(size_bin(15.01), "gt_15mm")
        self.assertAlmostEqual(esd_mm(4.0 * math.pi * 5.0**3 / 3.0), 10.0)

    def test_detection_false_positives_and_dice(self) -> None:
        gt = np.zeros((30, 30, 30), dtype=bool)
        gt[5:8, 5:8, 5:8] = True          # 27 voxels, ESD 3.7 mm -> 0-5 bin
        gt[15:25, 15:25, 15:25] = True    # 1000 voxels, ESD 12.4 mm -> 10-15 bin
        pred = np.zeros_like(gt)
        pred[15:25, 15:25, 15:25] = True  # hits the large lesion exactly
        pred[1:3, 26:28, 26:28] = True    # false positive
        result = evaluate_case(gt, pred, 1.0, 0.10)
        by_bin = {l["size_bin"]: l for l in result["lesions"]}
        self.assertFalse(by_bin["0_5mm"]["detected"])
        self.assertTrue(by_bin["10_15mm"]["detected"])
        self.assertAlmostEqual(by_bin["10_15mm"]["lesion_dice"], 1.0)
        self.assertEqual(len(result["false_positives"]), 1)
        agg = aggregate([result], 0.10)
        self.assertEqual(agg["bins"]["0_5mm"]["sensitivity"], 0.0)
        self.assertEqual(agg["bins"]["10_15mm"]["sensitivity"], 1.0)
        self.assertIsNone(agg["bins"]["gt_15mm"]["sensitivity"])
        lo, hi = wilson(9, 10)
        self.assertTrue(0.55 < lo < 0.9 < hi <= 1.0)


if __name__ == "__main__":
    unittest.main()
