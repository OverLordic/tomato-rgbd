"""ZIP integrity and extraction boundaries, using tiny local archives."""

import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from tomato_rgbd.download import extract_archive


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "sample.zip"
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.receipt = {"sha256": "synthetic-archive"}

    def test_complete_extraction_is_reused_with_crc_receipt(self):
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr("single_plant_cam/train/sample.txt", "sample")
        out = extract_archive(self.archive, self.raw, self.receipt)
        self.assertEqual((out / "train/sample.txt").read_text(), "sample")
        self.assertTrue(
            json.loads((self.raw / "extraction_receipt.json").read_text())[
                "zip_crc_checked"
            ]
        )
        self.assertEqual(extract_archive(self.archive, self.raw, self.receipt), out)

    def test_parent_path_cannot_escape_raw_directory(self):
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr("single_plant_cam/../../escape.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Unexpected ZIP"):
            extract_archive(self.archive, self.raw, self.receipt)
        self.assertFalse((self.root / "escape.txt").exists())

    def test_symlink_members_are_rejected(self):
        member = zipfile.ZipInfo("single_plant_cam/link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr(member, "../../outside")
        with self.assertRaisesRegex(ValueError, "Unexpected ZIP"):
            extract_archive(self.archive, self.raw, self.receipt)

    def test_existing_unverified_directory_is_not_overwritten(self):
        with zipfile.ZipFile(self.archive, "w") as z:
            z.writestr("single_plant_cam/value.txt", "new")
        out = self.raw / "single_plant_cam"
        out.mkdir()
        (out / "value.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "no matching"):
            extract_archive(self.archive, self.raw, self.receipt)
        self.assertEqual((out / "value.txt").read_text(), "keep")

    def test_corrupted_member_fails_crc_before_publishing_extraction(self):
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("single_plant_cam/value.txt", b"unique-content-123")
        content = self.archive.read_bytes()
        self.archive.write_bytes(
            content.replace(b"unique-content-123", b"unique-content-124")
        )
        with self.assertRaises(zipfile.BadZipFile):
            extract_archive(self.archive, self.raw, self.receipt)
        self.assertFalse((self.raw / "single_plant_cam").exists())


if __name__ == "__main__":
    unittest.main()
