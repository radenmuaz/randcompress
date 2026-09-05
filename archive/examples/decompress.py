"""
Decode a range-coded bundle → original file.

Usage:
    uv run python examples/decompress.py \
        --bundle runs/compressed \
        --output /tmp/recovered.bin \
        --verify datasets/surat_al-fatihah.txt
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from randcompress.decompress import decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="bundle dir (from compress.py)")
    p.add_argument("--output", required=True, help="output file path")
    p.add_argument("--verify", default=None,  help="original file to verify against")
    args = p.parse_args()

    decode(args.bundle, args.output, args.verify)


if __name__ == "__main__":
    main()
