"""Download and verify the official single-plant archive before extracting it."""

import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

URL = "https://data.4tu.nl/file/e1ee9de9-6ce9-4502-a37c-34b5b1372bed/923e3ab0-011f-432c-9039-2151e2c4a6f9"
ARCHIVE_SIZE = 7_456_535_483
OFFICIAL_MD5 = "81d46dddbddb6519736fedfa8d73f694"


def download_archive(path: Path) -> None:
    """Resume an interrupted transfer with curl. Existing complete files are checked later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == ARCHIVE_SIZE:
        return
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--progress-bar",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(path),
            URL,
        ],
        check=True,
    )


def verify_archive(path: Path) -> dict:
    """Read the entire archive once and compare its size and MD5 to the publisher."""
    if path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError(f"Incomplete archive: expected {ARCHIVE_SIZE} bytes")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    if md5.hexdigest() != OFFICIAL_MD5:
        raise ValueError(f"Archive MD5 differs from 4TU: {md5.hexdigest()}")
    return {
        "bytes": path.stat().st_size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "url": URL,
        "publisher_md5_matched": True,
    }


def extract_archive(path: Path, raw_dir: Path, receipt: dict) -> Path:
    """Verify ZIP CRCs while extracting, then publish the complete directory atomically.

    A completed extraction is reused on subsequent runs. Unexpected paths, symlinks,
    and an existing directory without a matching receipt are rejected.
    """
    destination = raw_dir / "single_plant_cam"
    marker = raw_dir / "extraction_receipt.json"
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = Path(member.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "single_plant_cam"
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unexpected ZIP member: {member.filename}")
        if destination.exists():
            if (
                not marker.exists()
                or json.loads(marker.read_text())["sha256"] != receipt["sha256"]
            ):
                raise ValueError("Existing extraction has no matching archive receipt")
            for member in members:
                if not member.is_dir():
                    target = raw_dir / member.filename
                    if (
                        not target.is_file()
                        or target.stat().st_size != member.file_size
                    ):
                        raise ValueError(
                            f"Incomplete extracted file: {member.filename}"
                        )
            return destination

        staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=raw_dir))
        try:
            # ZipFile validates each member's CRC while its contents are read
            archive.extractall(staging)
            (staging / "single_plant_cam").replace(destination)
            marker.write_text(
                json.dumps(
                    {**receipt, "zip_members": len(members), "zip_crc_checked": True},
                    indent=2,
                )
                + "\n"
            )
        finally:
            shutil.rmtree(staging)
    return destination
