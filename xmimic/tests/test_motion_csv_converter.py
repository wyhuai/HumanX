import copy
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_lib.motion_utils.motion_csv_converter import (
    _get_motion_entry,
    _default_import_pkl_path,
    export_motion_to_csv,
    import_csv_to_motion_file,
)


class MotionCsvConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.robot_config = cls.repo_root / "humanoidverse/config/robot/g1/g1_29dof_model16.yaml"
        cls.sample_pkl = cls.repo_root / "data/motions/humanx_demo/BMaster_catch_yaw0_mean.pkl"
        cls.sample_data = joblib.load(cls.sample_pkl)
        cls.motion_key = next(iter(cls.sample_data))

    def test_export_and_import_roundtrip_robot_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "traj.csv"
            output_pkl = tmpdir / "roundtrip.pkl"

            exported = export_motion_to_csv(
                input_pkl=self.sample_pkl,
                motion_key=self.motion_key,
                output_csv=csv_path,
                robot_config=self.robot_config,
            )

            self.assertTrue(exported)
            self.assertTrue(csv_path.exists())

            csv = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            self.assertEqual(csv.shape[1], 36)
            self.assertEqual(csv.shape[0], self.sample_data[self.motion_key]["pose_aa"].shape[0])

            imported = import_csv_to_motion_file(
                input_pkl=self.sample_pkl,
                motion_key=self.motion_key,
                input_csv=csv_path,
                robot_config=self.robot_config,
                output_pkl=output_pkl,
                inplace=False,
            )

            self.assertTrue(imported)
            self.assertTrue(output_pkl.exists())

            out = joblib.load(output_pkl)[self.motion_key]
            src = self.sample_data[self.motion_key]
            self.assertEqual(out["pose_aa"].shape[1:], (30, 3))
            np.testing.assert_allclose(out["root_trans_offset"], src["root_trans_offset"], atol=1e-6)
            np.testing.assert_allclose(out["root_rot"], src["root_rot"], atol=1e-6)
            np.testing.assert_allclose(out["dof"].reshape(csv.shape[0], -1), src["dof"].reshape(csv.shape[0], -1), atol=1e-6)
            np.testing.assert_allclose(out["pose_aa"][:, :30, :], src["pose_aa"][:, :30, :], atol=1e-5)

    def test_missing_motion_key_warns_and_skips_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "missing.csv"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                exported = export_motion_to_csv(
                    input_pkl=self.sample_pkl,
                    motion_key="does_not_exist",
                    output_csv=csv_path,
                    robot_config=self.robot_config,
                )
            self.assertFalse(exported)
            self.assertFalse(csv_path.exists())
            self.assertTrue(any("does_not_exist" in str(w.message) for w in caught))

    def test_single_key_defaults_to_first_motion(self):
        motion = _get_motion_entry(self.sample_data, None)
        self.assertEqual(motion, self.sample_data[self.motion_key])

    def test_multi_key_without_motion_key_raises_clear_error(self):
        multi = {
            "a": self.sample_data[self.motion_key],
            "b": copy.deepcopy(self.sample_data[self.motion_key]),
        }
        with self.assertRaisesRegex(ValueError, "multiple motion entries"):
            _get_motion_entry(multi, None)

    def test_default_import_output_appends_modified_suffix(self):
        path = Path("/tmp/motion_000096.pkl")
        self.assertEqual(
            _default_import_pkl_path(path),
            Path("/tmp/motion_000096_modified.pkl"),
        )

    def test_import_rejects_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "bad_rows.csv"
            output_pkl = tmpdir / "bad_rows.pkl"
            np.savetxt(csv_path, np.zeros((3, 36), dtype=np.float32), delimiter=",")

            with self.assertRaisesRegex(ValueError, "row count"):
                import_csv_to_motion_file(
                    input_pkl=self.sample_pkl,
                    motion_key=self.motion_key,
                    input_csv=csv_path,
                    robot_config=self.robot_config,
                    output_pkl=output_pkl,
                    inplace=False,
                )

            self.assertFalse(output_pkl.exists())


if __name__ == "__main__":
    unittest.main()
