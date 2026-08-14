import hashlib
import ast
import json
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "checkpoints/pretrained/manifest.json"
IDS_PATH = REPO_ROOT / "tools/assets/official_beat2_test_ids.json"
AESK_PATH = REPO_ROOT / "Scripts/FM/EMAGE_VAE/AESKConv_240_100.bin"
AESK_SHA256 = "5cd9566b24264f34d44003b3de62cdfd50aa85b7cdde2d369214599023c40f55"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_describes_two_unique_files_and_three_roles(self):
        self.assertEqual(self.manifest["format"], "streamtalk-retrained-cfg-v1")
        self.assertEqual(self.manifest["parameter_count"], 71_167_501)
        self.assertEqual(self.manifest["state_dict"]["tensor_count"], 407)
        self.assertEqual(self.manifest["evaluation"]["cfg_scale"], 3.0)
        self.assertEqual(self.manifest["evaluation"]["speaker2_clip_count"], 15)
        self.assertEqual(self.manifest["evaluation"]["all_clip_count"], 265)
        protocol = self.manifest["evaluation"]["protocol"].lower()
        self.assertIn("accelerated", protocol)
        self.assertIn("not been accepted", protocol)

        artifacts = self.manifest["artifacts"]
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(
            {role for artifact in artifacts for role in artifact["roles"]},
            {"speaker2", "speaker_all", "combined"},
        )
        self.assertEqual(len({artifact["file"] for artifact in artifacts}), 2)
        for artifact in artifacts:
            self.assertEqual(artifact["cfg_scale"], 3.0)
            self.assertEqual(artifact["size_bytes"], 284_796_631)
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")

    def test_sha256s_matches_manifest(self):
        checksum_lines = (
            REPO_ROOT / "checkpoints/pretrained/SHA256SUMS"
        ).read_text(encoding="utf-8").splitlines()
        checksums = {}
        for line in checksum_lines:
            digest, filename = line.split(maxsplit=1)
            checksums[filename] = digest
        expected = {
            artifact["file"]: artifact["sha256"]
            for artifact in self.manifest["artifacts"]
        }
        self.assertEqual(checksums, expected)

    def test_local_weights_match_manifest_when_present(self):
        weights_dir = MANIFEST_PATH.parent
        existing = [
            weights_dir / artifact["file"]
            for artifact in self.manifest["artifacts"]
            if (weights_dir / artifact["file"]).is_file()
        ]
        if not existing:
            self.skipTest("release weights are intentionally not stored in Git")
        self.assertEqual(len(existing), len(self.manifest["artifacts"]))
        for artifact in self.manifest["artifacts"]:
            path = weights_dir / artifact["file"]
            with self.subTest(weight=path.name):
                self.assertEqual(path.stat().st_size, artifact["size_bytes"])
                self.assertEqual(sha256_file(path), artifact["sha256"])

    def test_official_id_fixture_counts_and_speaker_subset(self):
        ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(ids), {"all", "speaker2"})
        self.assertEqual(len(ids["all"]), 265)
        self.assertEqual(len(set(ids["all"])), 265)
        self.assertEqual(len(ids["speaker2"]), 15)
        self.assertEqual(len(set(ids["speaker2"])), 15)
        self.assertEqual(
            ids["speaker2"],
            [file_id for file_id in ids["all"] if file_id.startswith("2_scott_0_")],
        )

    def test_aesk_asset_matches_pinned_hash(self):
        self.assertEqual(sha256_file(AESK_PATH), AESK_SHA256)

    def test_readme_exposes_release_and_reproduction_paths(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        required = (
            "## 💖 Inference Data",
            "## 🧠 Pretrained Models",
            "streamtalk_speaker2_combined_e0946_cfg3.pt",
            "streamtalk_speaker_all_e0940_cfg3.pt",
            "TestFixedExpressions_keyposes_all.py",
            "evaluate_generated_fgd.py",
            "--cfg_scale 3",
            "https://huggingface.co/X-Zhang/StreamTalk",
            "https://huggingface.co/datasets/X-Zhang/StreamTalk-Inference-Data",
            "https://huggingface.co/datasets/H-Liu1997/BEAT2",
            "hf download H-Liu1997/BEAT2",
            '--include "beat_english_v2.0.0/**"',
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)
        self.assertIsNone(re.search(r"calibrated.{0,80}oracle", readme, re.I | re.S))

        inference_section = readme.split("## 💖 Inference Data", 1)[1].split(
            "## 🧠 Pretrained Models", 1
        )[0]
        pretrained_section = readme.split("## 🧠 Pretrained Models", 1)[1].split(
            "## 💡 Overview", 1
        )[0]
        for section in (inference_section, pretrained_section):
            self.assertNotIn("FGD", section)
            self.assertNotRegex(section, r"0\.(?:378|217|989)")

        news_section = readme.split("## 📣 News", 1)[1].split(
            "## 💖 Inference Data", 1
        )[0]
        self.assertEqual(news_section.count("- **["), 1)
        for removed_text in (
            "## 🛠️ Troubleshooting",
            "--face_checkpoint",
            "SimpleSpeechModel",
            "This is the upstream baseline training recipe",
            "The generator files exceed GitHub's per-file limit",
            "StreamTalk generator checkpoints",
        ):
            with self.subTest(removed_text=removed_text):
                self.assertNotIn(removed_text, readme)

    def test_public_inference_has_no_private_asset_defaults(self):
        single_path = REPO_ROOT / "Scripts/FM/TestFixedExpressions_keyposes.py"
        nearest_path = REPO_ROOT / "Scripts/FM/FindNearest.py"
        data_builder_path = REPO_ROOT / "Data/BEAT2/create_h5.py"
        single_source = single_path.read_text(encoding="utf-8")
        nearest_source = nearest_path.read_text(encoding="utf-8")
        data_builder_source = data_builder_path.read_text(encoding="utf-8")
        self.assertNotIn("/mnt/4TDisk", single_source)
        self.assertNotIn("/mnt/4TDisk", nearest_source)
        self.assertNotIn("/mnt/4TDisk", data_builder_source)
        self.assertIn("--wavlm_model", data_builder_source)

        arguments = {}
        for node in ast.walk(ast.parse(single_source)):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if isinstance(node.args[0], ast.Constant):
                arguments[node.args[0].value] = {
                    keyword.arg: keyword.value for keyword in node.keywords
                }
        for flag in (
            "--wavlm_model",
            "--smplx_model",
            "--retrieval_data_folder",
            "--retrieval_csv_file",
        ):
            required = arguments[flag]["required"]
            self.assertIsInstance(required, ast.Constant)
            self.assertIs(required.value, True)


if __name__ == "__main__":
    unittest.main()
