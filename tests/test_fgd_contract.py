from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from streamtalk_fgd import ContractError, per_clip_truncated_lengths


class FGDContractTests(unittest.TestCase):
    def test_per_clip_truncation_preserves_boundaries(self):
        self.assertEqual(per_clip_truncated_lengths([63, 64, 97]), (32, 64, 96))

    def test_short_clip_is_rejected(self):
        with self.assertRaises(ContractError):
            per_clip_truncated_lengths([31])

    def test_invalid_multiple_is_rejected(self):
        with self.assertRaises(ValueError):
            per_clip_truncated_lengths([64], multiple=0)


if __name__ == "__main__":
    unittest.main()
