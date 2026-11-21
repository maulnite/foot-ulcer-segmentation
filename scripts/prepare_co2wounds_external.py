"""Bounded acquisition of the official, version-pinned CO2Wounds Mendeley ZIP."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402

URL = "https://data.mendeley.com/public-api/zip/s2w7rjwz49/download/1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "preregistration.json").read_text())
    archive = ROOT / "data/cache/co2wounds_v2_mendeley_v1.zip"
    target = ROOT / "data/raw/co2wounds_v2_mendeley_v1"
    metadata = ROOT / "data/metadata/co2wounds_v2_mendeley_v1_acquisition.json"
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if digest_file(archive) != record["archive_sha256"] or not target.is_dir():
            raise ValueError("Existing acquisition changed; preserve it and investigate")
        print(json.dumps({"status": "cached", **record}, indent=2))
        return
    if archive.exists() or target.exists():
        raise FileExistsError("Inspect incomplete acquisition before resuming")
    archive.parent.mkdir(exist_ok=True, parents=True)
    partial = archive.with_suffix(".zip.part")
    request = urllib.request.Request(URL, headers={"User-Agent": "CP2 academic evaluation"})
    digest, downloaded = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared > protocol["acquisition_limit_bytes"]:
            raise ValueError("Download exceeds preregistered size limit")
        headers = dict(response.headers)
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            if downloaded == 0 and not block.startswith(b"PK\x03\x04"):
                raise ValueError("Response is not a ZIP archive")
            downloaded += len(block)
            if downloaded > protocol["acquisition_limit_bytes"]:
                raise ValueError("Download exceeded bounded acquisition")
            stream.write(block)
            digest.update(block)
        if declared and downloaded != declared:
            raise ValueError("Truncated archive response")
    with zipfile.ZipFile(partial) as source:
        members = source.infolist()
        size = sum(member.file_size for member in members)
        if size > protocol["extraction_limit_bytes"] or len(members) > 10000:
            raise ValueError("Extraction exceeds preregistered limit")
        for member in members:
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts or (member.external_attr >> 16) & 0xF000 == 0xA000:
                raise ValueError("Unsafe archive member")
        damaged = source.testzip()
        if damaged:
            raise ValueError(f"Archive CRC failure: {damaged}")
        staging = target.with_name(target.name + ".extracting")
        staging.mkdir(parents=True, exist_ok=False)
        for member in members:
            destination = staging / member.filename
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        staging.rename(target)
    partial.rename(archive)
    record = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "url": URL, "doi": "10.17632/s2w7rjwz49.1", "version": 1,
              "archive_bytes": downloaded, "archive_sha256": digest.hexdigest(),
              "uncompressed_bytes": size, "members": len(members), "zip_crc_verified": True,
              "provider_checksum_available": False, "http_headers": headers,
              "dataset_root": str(target.relative_to(ROOT)),
              "license": "Mendeley version1: CC BY 4.0; author GitHub README: CC BY-NC-ND; discrepancy retained",
              "usage": "local noncommercial academic evaluation; no raw media redistribution"}
    write_json(metadata, record)
    write_json(args.run / "acquisition.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
