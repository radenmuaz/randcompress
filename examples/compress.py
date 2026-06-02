"""
Encode a file using trained adapter weights → range-coded bundle.

Usage:
    uv run python examples/compress.py \
        --ckpt runs/train_msrnn/ckpt_last \
        --input datasets/surat_al-fatihah.txt \
        --output runs/compressed
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from randcompress.checkpoint import load_checkpoint
from randcompress.compress import encode
from randcompress.models import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="checkpoint dir (from training)")
    p.add_argument("--input",  required=True, help="original file to compress")
    p.add_argument("--output", required=True, help="output bundle dir")
    args = p.parse_args()

    mcfg, tcfg, adapters = load_checkpoint(args.ckpt)
    model  = get_model(mcfg, tcfg)
    frozen = model.init_frozen(mcfg.seed)

    with open(args.input, "rb") as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8).copy()

    encode(model, frozen, adapters, mcfg, tcfg, raw_bytes, args.output)


if __name__ == "__main__":
    main()
