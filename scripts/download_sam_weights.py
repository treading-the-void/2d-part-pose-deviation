"""Download the official SAM ViT-B checkpoint used by this project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "sam_vit_b_01ec64.pth"
MIN_BYTES = 100 * 1024 * 1024  # Reject an HTML error page or an incomplete checkpoint.
CHUNK_BYTES = 1024 * 1024


def download(destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        if destination.stat().st_size < MIN_BYTES:
            raise ValueError(f"Existing file is too small: {destination}; rerun with --force")
        print(f"Already present: {destination} ({destination.stat().st_size:,} bytes)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = Request(URL, headers={"User-Agent": "2d-part-pose-deviation/1.0"})
    try:
        with urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            expected = response.headers.get("Content-Length")
            expected_bytes = int(expected) if expected and expected.isdigit() else None
            if expected_bytes is not None and expected_bytes < MIN_BYTES:
                raise ValueError(f"Server returned only {expected_bytes:,} bytes; not a SAM checkpoint")
            downloaded = 0
            next_report = 16 * CHUNK_BYTES
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if expected_bytes:
                        print(f"Downloaded {downloaded / expected_bytes:.0%} ({downloaded:,} bytes)")
                    else:
                        print(f"Downloaded {downloaded:,} bytes")
                    next_report += 16 * CHUNK_BYTES
        if downloaded < MIN_BYTES:
            raise ValueError(f"Download is too small: {downloaded:,} bytes")
        if expected_bytes is not None and downloaded != expected_bytes:
            raise ValueError(f"Incomplete download: {downloaded:,} of {expected_bytes:,} bytes")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {destination} ({downloaded:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official SAM ViT-B checkpoint")
    parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint")
    args = parser.parse_args()
    try:
        download(DESTINATION, force=args.force)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"SAM download failed: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
