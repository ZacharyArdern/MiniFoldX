"""MiniFold-MLX command-line interface.

Usage
-----
    minifold sequences.fasta --out_dir ./structures
    minifold sequences.fasta --out_dir ./structures --model_size 12L --no_compile
    minifold sequences.fasta --out_dir ./structures --recycling 1
    minifold --example

Weights are downloaded automatically from HuggingFace on first run
and cached in ~/.cache/huggingface/hub/.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")

import platform
if platform.system() != "Darwin":
    print("ERROR: MiniFold-MLX requires macOS with Apple Silicon. "
          "For Linux/Windows, use the original MiniFold: https://github.com/jwohlwend/minifold")
    sys.exit(1)

import time
from pathlib import Path


HF_REPO = "z-ardern/MiniFoldX_weights"

EXAMPLE_ID  = "7L8N_A"
EXAMPLE_SEQ = ("QQLSDDEKLVAAFVKAVAYMSPRKIGALVSIEETQTLREYIATGIPLDADISGELLINIFIPNTPLHDGAVIVEGNKIAVSC"
               "AYLPLSESSHISKEFGTRHRAAIGLSEASDAFTFVVSEETGAISVAYKGDFIHDLSLEAFEVLLREHFI")


def _get_weights(model_size: str, use_quantized_esm: bool = True) -> tuple[str, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    esm_folder  = "ESM2_MiniFold_int8" if use_quantized_esm else "ESM2_MiniFold"
    skip_folder = "ESM2_MiniFold"      if use_quantized_esm else "ESM2_MiniFold_int8"
    print(f"Checking weights ({HF_REPO}) …")
    weights_dir   = Path(snapshot_download(HF_REPO, ignore_patterns=[f"{skip_folder}/*"]))
    esm_path      = weights_dir / esm_folder
    minifold_path = weights_dir / f"minifold_{model_size}"

    if not esm_path.exists():
        print(f"ERROR: {esm_folder}/ not found in {weights_dir}")
        sys.exit(1)
    if not minifold_path.exists():
        print(f"ERROR: minifold_{model_size}/ not found in {weights_dir}")
        sys.exit(1)

    return str(esm_path), str(minifold_path)


def _read_fasta(path: str) -> list[tuple[str, str]]:
    import gzip
    sequences = []
    current_id, current_seq = None, []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))
    return sequences


def _run(sequences, args, out_dir):
    use_quantized_esm = not args.non_quantized_esm2
    esm_path, minifold_path = _get_weights(args.model_size, use_quantized_esm)

    from minifoldx import load_model, predict_sequence

    print(f"\nLoading model ({args.model_size}) …")
    tokenizer, model = load_model(esm_path, minifold_path, bf16=False, fp16=False)
    model.convert_to_bf16()
    model.enable_sgmm_gate()
    model.fold._timing = False

    if use_quantized_esm:
        print("  ESM2: loaded pre-quantized int8 weights")

    if not args.no_compile:
        model.enable_compile_miniformer()

    print(f"\nPredicting {len(sequences)} sequence(s) → {out_dir}/\n")
    t_start = time.perf_counter()

    for seq_id, seq in sequences:
        t0 = time.perf_counter()
        pdb_str = predict_sequence(
            seq_id, seq, model, tokenizer,
            num_recycling=args.recycling,
            max_seq_len=args.max_len,
            early_stop=args.early_stop,
        )
        elapsed = time.perf_counter() - t0

        if pdb_str is None:
            print(f"  SKIP  {seq_id}  (len={len(seq)})")
            continue

        out_path = out_dir / f"{seq_id}.pdb"
        out_path.write_text(pdb_str)
        print(f"  {seq_id}  len={len(seq)}  {elapsed:.2f}s  → {out_path}")

    total = time.perf_counter() - t_start
    print(f"\nDone. {len(sequences)} sequence(s) in {total:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="MiniFold-MLX structure prediction")
    parser.add_argument("fasta", nargs="?", help="Input FASTA file")
    parser.add_argument("--example", action="store_true",
                        help=f"Fold the built-in example sequence ({EXAMPLE_ID}) and exit")
    parser.add_argument("--out_dir", default="./structures",
                        help="Output directory for PDB files (default: ./structures)")
    parser.add_argument("--model_size", choices=["48L", "12L"], default="48L",
                        help="MiniFold model size (default: 48L)")
    parser.add_argument("--recycling", type=int, default=0,
                        help="Number of recycling iterations (default: 0)")
    parser.add_argument("--max_len", type=int, default=800,
                        help="Skip sequences longer than this (default: 800, 0=no limit)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable mx.compile on MiniFormer")
    parser.add_argument("--non-quantized-ESM2", dest="non_quantized_esm2",
                        action="store_true",
                        help="Use full-precision ESM2 weights instead of int8 (~11 GB vs ~5.5 GB)")
    parser.add_argument("--early-stop", dest="early_stop", type=float, default=None,
                        metavar="PLDDT",
                        help="Stop recycling early when mean pLDDT >= this value (e.g. 85)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.example:
        print(f"Running example sequence: {EXAMPLE_ID} (len={len(EXAMPLE_SEQ)})")
        _run([(EXAMPLE_ID, EXAMPLE_SEQ)], args, out_dir)
        return

    if not args.fasta:
        parser.error("a FASTA file is required (or use --example)")

    sequences = _read_fasta(args.fasta)
    if not sequences:
        print(f"ERROR: no sequences found in {args.fasta}")
        sys.exit(1)
    print(f"Found {len(sequences)} sequence(s) in {args.fasta}")

    _run(sequences, args, out_dir)


if __name__ == "__main__":
    main()
