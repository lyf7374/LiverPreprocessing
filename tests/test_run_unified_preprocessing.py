from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.run_unified_preprocessing import (
    build_stage_commands,
    discover_waw_four_phase_patients,
    resolve_dataset_roots,
    write_waw_selection,
)


class UnifiedPreprocessingRunnerTests(unittest.TestCase):
    def test_discovers_only_complete_four_phase_waw_cases_with_one_annotated_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            waw_root = Path(temporary)
            with (waw_root / "ct_hcc_metadata.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["patient_id", "ct_phase", "ct_file_name"]
                )
                writer.writeheader()
                for patient_id, phases in (("2", range(4)), ("3", range(3)), ("4", range(4))):
                    for phase in phases:
                        writer.writerow(
                            {
                                "patient_id": patient_id,
                                "ct_phase": str(phase),
                                "ct_file_name": f"{patient_id}_{phase}",
                            }
                        )
            with zipfile.ZipFile(waw_root / "tumor_masks.zip", "w") as archive:
                archive.writestr("masks/2_1_0_tumor_seg.nrrd", b"mask")
                archive.writestr("masks/4_1_0_tumor_seg.nrrd", b"mask")
                archive.writestr("masks/4_2_0_tumor_seg.nrrd", b"mask")

            selected = discover_waw_four_phase_patients(waw_root)

        self.assertEqual(selected, ["2"])

    def test_builds_geometry_conversion_and_label_stages_in_order(self) -> None:
        roots = {"PLC-CECT": Path("D:/raw/PLC-CECT"), "WAW-TACE": Path("D:/raw/WAW-TACE"), "MCT-LTDiag": Path("D:/raw/MCT-LTDiag")}
        commands = build_stage_commands(
            python_executable=Path("C:/Python/python.exe"),
            project_root=Path("C:/project"),
            dataset_roots=roots,
            output_root=Path("D:/processed/unified"),
            waw_selection=Path("D:/processed/unified/manifests/waw_four_phase_patients.csv"),
            workers=4,
            replace_datasets=("PLC-CECT", "WAW-TACE"),
        )

        self.assertEqual(len(commands), 3)
        self.assertTrue(str(commands[0][1]).endswith("estimate_plc_geometry.py"))
        self.assertEqual(commands[0][2], "all")
        self.assertEqual(commands[0][commands[0].index("--plc-root") + 1], str(roots["PLC-CECT"]))
        self.assertTrue(str(commands[1][1]).endswith("preprocess_multiphase_ct.py"))
        self.assertTrue(str(commands[2][1]).endswith("prepare_hierarchical_labels.py"))
        self.assertIn("--plc-geometry", commands[1])
        self.assertIn("--mct-root", commands[1])
        self.assertIn("--replace-datasets", commands[1])
        self.assertNotIn("--replace-datasets", commands[2])
        self.assertEqual(commands[2][-2:], ["--root", str(Path("D:/processed/unified"))])

    def test_omits_geometry_stage_when_reusing_an_existing_csv(self) -> None:
        commands = build_stage_commands(
            python_executable=Path("C:/Python/python.exe"),
            project_root=Path("C:/project"),
            dataset_roots={"PLC-CECT": Path("D:/raw/PLC-CECT")},
            output_root=Path("D:/processed/unified"),
            waw_selection=Path("D:/processed/unified/manifests/waw_four_phase_patients.csv"),
            workers=4,
            datasets=("PLC-CECT",),
            include_geometry_stage=False,
        )

        self.assertEqual(len(commands), 2)
        self.assertTrue(str(commands[0][1]).endswith("preprocess_multiphase_ct.py"))

    def test_skips_geometry_stage_when_plc_is_not_selected(self) -> None:
        commands = build_stage_commands(
            python_executable=Path("C:/Python/python.exe"),
            project_root=Path("C:/project"),
            dataset_roots={"MCT-LTDiag": Path("D:/raw/MCT-LTDiag")},
            output_root=Path("D:/processed/unified"),
            waw_selection=Path("D:/processed/unified/manifests/waw_four_phase_patients.csv"),
            workers=4,
            datasets=("MCT-LTDiag",),
        )

        self.assertEqual(len(commands), 2)
        self.assertTrue(str(commands[0][1]).endswith("preprocess_multiphase_ct.py"))
        self.assertEqual(commands[0][commands[0].index("--datasets") + 1 :], ["MCT-LTDiag", "--mct-root", str(Path("D:/raw/MCT-LTDiag"))])
        self.assertNotIn("--plc-root", commands[0])

    def test_resolves_explicit_roots_over_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / "MCT-LTDiag").mkdir()
            elsewhere = raw / "plc_elsewhere"
            elsewhere.mkdir()
            roots = resolve_dataset_roots(("MCT-LTDiag", "PLC-CECT"), raw, {"PLC-CECT": elsewhere})
            self.assertEqual(roots["MCT-LTDiag"], raw / "MCT-LTDiag")
            self.assertEqual(roots["PLC-CECT"], elsewhere)
            with self.assertRaises(SystemExit):
                resolve_dataset_roots(("WAW-TACE",), raw, {})

    def test_refuses_to_write_an_empty_waw_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.csv"

            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                write_waw_selection(path, [])

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
