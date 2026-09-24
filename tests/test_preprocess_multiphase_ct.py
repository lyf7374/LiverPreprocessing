from __future__ import annotations

import unittest

import numpy as np

import SimpleITK as sitk  # noqa: E402

from scripts.preprocess_multiphase_ct import (  # noqa: E402
    PLC_IN_PLANE_SPACING_MM,
    apply_phase_geometry,
    flip_k_axis,
    isotropic_output_grid,
    plc_uint8_to_hu,
    to_int16_hu,
)


class PlcConventionTests(unittest.TestCase):
    def test_uint8_window_maps_back_to_hounsfield_units(self) -> None:
        image = sitk.GetImageFromArray(np.array([[[0, 127.5, 255]]], dtype=np.float32))
        hu = sitk.GetArrayFromImage(plc_uint8_to_hu(image))
        np.testing.assert_allclose(hu, [[[-200.0, 0.0, 200.0]]])

    def test_flip_reverses_slice_order_on_the_same_grid(self) -> None:
        array = np.arange(24, dtype=np.float32).reshape(4, 3, 2)  # (z, y, x)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.5, 0.5, 2.0))
        image.SetOrigin((1.0, 2.0, 3.0))
        flipped = flip_k_axis(image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(flipped), array[::-1])
        self.assertEqual(flipped.GetSpacing(), image.GetSpacing())
        self.assertEqual(flipped.GetOrigin(), image.GetOrigin())

    def test_phase_geometry_sets_spacing_and_optionally_flips(self) -> None:
        image = sitk.GetImageFromArray(np.arange(8, dtype=np.uint8).reshape(2, 2, 2))
        geometry = {"spacing_mm": [PLC_IN_PLANE_SPACING_MM, PLC_IN_PLANE_SPACING_MM, 2.5], "k_axis_flip": True}
        result = apply_phase_geometry(image, geometry)
        self.assertEqual(result.GetSpacing(), (PLC_IN_PLANE_SPACING_MM, PLC_IN_PLANE_SPACING_MM, 2.5))
        self.assertEqual(sitk.GetArrayFromImage(result)[0, 0, 0], 4)


class OutputGridTests(unittest.TestCase):
    def test_one_millimetre_grid_covers_the_crop_extent(self) -> None:
        reference = sitk.Image((100, 80, 40), sitk.sitkFloat32)
        reference.SetSpacing((0.78125, 0.78125, 2.5))
        reference.SetOrigin((-200.0, -150.0, 10.0))
        grid = isotropic_output_grid(reference, start=[10, 20, 4], size=[64, 32, 8])
        self.assertEqual(grid.GetSpacing(), (1.0, 1.0, 1.0))
        np.testing.assert_allclose(grid.GetOrigin(), (-200.0 + 10 * 0.78125, -150.0 + 20 * 0.78125, 10.0 + 4 * 2.5))
        # extent of 63 * 0.78125 = 49.2 mm -> 50 voxels; 31 * 0.78125 = 24.2 -> 25; 7 * 2.5 = 17.5 -> 18
        self.assertEqual(grid.GetSize(), (50, 25, 18))

    def test_int16_output_rounds_hounsfield_units(self) -> None:
        image = sitk.GetImageFromArray(np.array([[[-1000.4, 39.6, 40000.0]]], dtype=np.float32))
        result = to_int16_hu(image)
        self.assertEqual(result.GetPixelIDTypeAsString(), "16-bit signed integer")
        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), [[[-1000, 40, 32767]]])


if __name__ == "__main__":
    unittest.main()
