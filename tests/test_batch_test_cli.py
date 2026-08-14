import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Scripts/FM/TestFixedExpressions_keyposes_all.py"
SPEC = importlib.util.spec_from_file_location("streamtalk_batch_test", SCRIPT_PATH)
BATCH_TEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH_TEST)


class BatchTestCliTests(unittest.TestCase):
    def test_auxiliary_checkpoint_is_internal_not_a_public_cli_option(self):
        batch_source = SCRIPT_PATH.read_text(encoding="utf-8")
        single_source = SCRIPT_PATH.with_name(
            "TestFixedExpressions_keyposes.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--face_checkpoint", batch_source)
        self.assertNotIn("--face_checkpoint", single_source)
        self.assertIn("BUNDLED_FACE_CHECKPOINT", single_source)
        self.assertIn("ckpt/split/SimpleSpeechModel/best.pt", single_source)

    def test_speaker2_dry_run_builds_complete_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            motion_dir = root / "motion"
            wav_dir = root / "wav"
            output_dir = root / "output"
            motion_dir.mkdir()
            wav_dir.mkdir()
            file_id = "2_scott_0_1_1"
            (motion_dir / f"{file_id}.npz").touch()
            (wav_dir / f"{file_id}.wav").touch()
            csv_path = root / "split.csv"
            csv_path.write_text(
                "id,type\n"
                "10_kieks_0_1_1,test\n"
                f"{file_id},test\n"
                "2_scott_0_2_2,train\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                BATCH_TEST.Test_All(
                    [
                        "--data_folder",
                        str(motion_dir),
                        "--wav_folder",
                        str(wav_dir),
                        "--csv_file",
                        str(csv_path),
                        "--checkpoint",
                        str(root / "generator.pt"),
                        "--output_folder",
                        str(output_dir),
                        "--wavlm_model",
                        "local/wavlm",
                        "--smplx_model",
                        str(root / "SMPLX_NEUTRAL_2020.npz"),
                        "--pid",
                        "2_scott_0",
                        "--cfg_scale",
                        "3",
                        "--seed",
                        "7",
                        "--dry_run",
                    ]
                )

            rendered = stdout.getvalue()
            self.assertIn("TestFixedExpressions_keyposes.py", rendered)
            self.assertIn(str(motion_dir / f"{file_id}.npz"), rendered)
            self.assertIn(str(wav_dir / f"{file_id}.wav"), rendered)
            self.assertIn(str(output_dir / f"res_{file_id}.npz"), rendered)
            self.assertIn("--pid 2_scott_0", rendered)
            self.assertIn("--cfg_scale 3.0", rendered)
            self.assertIn("--seed 7", rendered)
            self.assertNotIn("--face_checkpoint", rendered)
            self.assertIn("completed 1 test files", rendered)
            self.assertTrue(output_dir.is_dir())

    def test_speaker_id_validation(self):
        self.assertEqual(BATCH_TEST._speaker_id("2_scott_0_1_1"), "2_scott_0")
        with self.assertRaises(ValueError):
            BATCH_TEST._speaker_id("malformed")


if __name__ == "__main__":
    unittest.main()
