import math
import unittest

from scripts.prepare_hierarchical_labels import (
    equivalent_spherical_diameter_mm,
    label_for_diagnosis,
    size_bin,
)


class HierarchicalLabelTests(unittest.TestCase):
    def test_maps_hcc_to_primary_malignancy(self):
        labels = label_for_diagnosis("HCC")
        self.assertEqual(labels["lesion_presence"], "present")
        self.assertEqual(labels["behavior"], "malignant")
        self.assertEqual(labels["origin"], "primary_hepatic")
        self.assertEqual(labels["diagnosis_coarse"], "HCC")

    def test_maps_crlm_to_metastasis_coarse_class(self):
        labels = label_for_diagnosis("CRLM")
        self.assertEqual(labels["origin"], "extrahepatic_metastatic")
        self.assertEqual(labels["diagnosis_coarse"], "metastasis")
        self.assertEqual(labels["diagnosis_fine"], "CRLM")

    def test_maps_control_to_absent_and_masked_hierarchies(self):
        labels = label_for_diagnosis("CN")
        self.assertEqual(labels["lesion_presence_id"], 0)
        self.assertEqual(labels["behavior_id"], -1)
        self.assertEqual(labels["origin_id"], -1)

    def test_size_bins_have_nonoverlapping_closed_upper_bounds(self):
        self.assertEqual(size_bin(5.0), "0-5")
        self.assertEqual(size_bin(5.0001), "5-10")
        self.assertEqual(size_bin(10.0), "5-10")
        self.assertEqual(size_bin(10.0001), "10-15")
        self.assertEqual(size_bin(15.0), "10-15")
        self.assertEqual(size_bin(15.0001), ">15")

    def test_equivalent_diameter_inverts_sphere_volume(self):
        radius_mm = 5.0
        volume_mm3 = 4.0 * math.pi * radius_mm**3 / 3.0
        self.assertAlmostEqual(equivalent_spherical_diameter_mm(volume_mm3), 10.0)


if __name__ == "__main__":
    unittest.main()
