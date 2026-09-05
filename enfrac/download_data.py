"""Download enwik8/enwik9 (Matt Mahoney's Large Text Compression Benchmark corpora) and unzip
them into datasets/. Streams to disk with a progress bar (files are 36MB/322MB zipped) and
verifies the extracted size against the well-known enwik8/enwik9 byte counts before deleting the
.zip, so a truncated/corrupt download fails loudly instead of leaving a bad file in datasets/.
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile

from tqdm import tqdm

SOURCES = {
    "enwik8": {
        "url": "https://mattmahoney.net/dc/enwik8.zip",
        "member": "enwik8",
        "n_bytes": 100_000_000,
    },
    "enwik9": {
        "url": "https://mattmahoney.net/dc/enwik9.zip",
        "member": "enwik9",
        "n_bytes": 1_000_000_000,
    },
}


def _download(url: str, dest: str) -> None:
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def fetch(name: str, out_dir: str, keep_zip: bool = False) -> str:
    spec = SOURCES[name]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name)
    zip_path = os.path.join(out_dir, f"{name}.zip")

    if os.path.exists(out_path) and os.path.getsize(out_path) == spec["n_bytes"]:
        print(f"{name}: already present at {out_path} ({spec['n_bytes']:,} B), skipping")
        return out_path

    print(f"{name}: downloading {spec['url']} -> {zip_path}")
    _download(spec["url"], zip_path)

    print(f"{name}: extracting {spec['member']} -> {out_path}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(spec["member"]) as src, open(out_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)

    n_bytes = os.path.getsize(out_path)
    if n_bytes != spec["n_bytes"]:
        raise RuntimeError(
            f"{name}: extracted size {n_bytes:,} B != expected {spec['n_bytes']:,} B "
            f"-- download likely truncated/corrupt, not deleting {zip_path}")

    if not keep_zip:
        os.remove(zip_path)
    print(f"{name}: OK, {n_bytes:,} B -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--which", default="enwik8,enwik9", help="comma-separated subset of enwik8,enwik9")
    p.add_argument("--out_dir", default="datasets")
    p.add_argument("--keep_zip", action="store_true", help="keep the .zip after extraction")
    args = p.parse_args()

    for name in args.which.split(","):
        name = name.strip()
        if name not in SOURCES:
            raise ValueError(f"unknown dataset {name!r}, expected one of {list(SOURCES)}")
        fetch(name, args.out_dir, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
