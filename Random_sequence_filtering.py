# -*- coding: utf-8 -*-
"""
High-throughput random DNA/RNA sequence generator with chunked, compressed output.

Features
--------
- Scales to billions of sequences via streaming (no need to keep all in memory)
- Chunked writing to multiple files for easier storage & parallel downstream use
- Optional GC-content pre-filter (min/max)
- Reproducible with --seed
- Supports FASTA (.fa/.fasta) and CSV (.csv) outputs, with optional gzip (.gz)
- Efficient NumPy-based generation with per-shard approximate de-dup (hash-based)

Examples
--------
# Generate 1e8 sequences of length 80, in 1e6-sized chunks (100 files), FASTA+gzip:
python generate_random_sequences.py \
  --num 100000000 --length 80 --chunk-size 1000000 \
  --outdir results_seqs --prefix rand80 \
  --format fasta --gzip \
  --gc-min 0.60 --gc-max 0.75 \
  --seed 42

# CSV output (easier to merge scores later):
python generate_random_sequences.py \
  --num 50000000 --length 80 \
  --chunk-size 2000000 \
  --outdir results_csv \
  --prefix rand80 --format csv --gzip

# Allow duplicates (faster, smaller memory):
python generate_random_sequences.py \
  --num 100000000 --length 80 --chunk-size 1000000 \
  --outdir results_fast --prefix rand80 \
  --format fasta --gzip --allow-duplicates
"""
import argparse
import csv
import gzip
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


ALPHABETS = {
    "ACGT": np.frombuffer(b"ACGT", dtype="S1"),  # A=0, C=1, G=2, T=3
    "ACGU": np.frombuffer(b"ACGU", dtype="S1"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-generate random DNA/RNA sequences at scale with optional GC filtering."
    )
    p.add_argument("--num", type=int, required=True,
                   help="Total number of sequences to generate (after filtering).")
    p.add_argument("--length", type=int, default=80,
                   help="Sequence length (default: 80).")
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="Number of sequences per output shard/file.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Generator batch size before filtering. "
                        "Default: 3x chunk-size (capped at 10M).")
    p.add_argument("--outdir", type=str, required=True,
                   help="Output directory.")
    p.add_argument("--prefix", type=str, default="seqs",
                   help="Output file prefix (default: seqs).")
    p.add_argument("--format", type=str, choices=["fasta", "csv"], default="fasta",
                   help="Output format (default: fasta).")
    p.add_argument("--gzip", action="store_true",
                   help="Compress each shard with gzip (.gz).")
    p.add_argument("--alphabet", type=str, choices=ALPHABETS.keys(), default="ACGT",
                   help="Alphabet to use (default: ACGT).")
    p.add_argument("--gc-min", type=float, default=None,
                   help="Minimum GC fraction to keep (e.g., 0.60).")
    p.add_argument("--gc-max", type=float, default=None,
                   help="Maximum GC fraction to keep (e.g., 0.75).")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed (default: 123).")
    p.add_argument("--start-index", type=int, default=1,
                   help="Starting index for sequence IDs (default: 1).")
    p.add_argument("--pad-width", type=int, default=12,
                   help="Zero-pad width for sequence IDs (default: 12).")
    p.add_argument("--allow-duplicates", action="store_true",
                   help="Allow duplicate sequences (default: False => "
                        "attempts to avoid duplicates per shard).")
    p.add_argument("--unique-scope", type=str, choices=["shard", "none"], default="shard",
                   help="Scope of uniqueness when duplicates are not allowed (default: shard).")
    return p.parse_args()


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_gc_frac(int_mat: np.ndarray) -> np.ndarray:
    """
    Given integer-encoded sequences (0..3), return GC fraction per sequence.
    A=0, C=1, G=2, T=3 => GC are 1 or 2.
    """
    is_gc = (int_mat == 1) | (int_mat == 2)
    gc_counts = is_gc.sum(axis=1)
    return gc_counts / int_mat.shape[1]


def ints_to_strs(int_mat: np.ndarray, alphabet_vec: np.ndarray) -> np.ndarray:
    """
    Convert integer-encoded sequences to unicode strings using the alphabet.
    Returns an array of dtype '<U{L}' strings.
    """
    bytes_arr = alphabet_vec[int_mat]  # (N, L) dtype=S1
    # Join each row of bytes into a string; generator keeps memory moderate.
    return np.fromiter(
        (b"".join(row.tolist()).decode("ascii") for row in bytes_arr),
        dtype=f"<U{int_mat.shape[1]}",
        count=int_mat.shape[0]
    )


def open_out(path: Path, use_gzip: bool):
    if use_gzip:
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    return open(path, "w", encoding="utf-8")


def write_fasta_shard(path: Path, ids: np.ndarray, seqs: np.ndarray, use_gzip: bool):
    with open_out(path, use_gzip) as f:
        for sid, s in zip(ids, seqs):
            f.write(f">{sid}\n{s}\n")


def write_csv_shard(path: Path, ids: np.ndarray, seqs: np.ndarray, use_gzip: bool):
    if use_gzip:
        fh = gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    else:
        fh = open(path, "w", encoding="utf-8", newline="")
    with fh as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence"])
        for sid, s in zip(ids, seqs):
            w.writerow([sid, s])


def make_ids(start: int, count: int, pad: int) -> np.ndarray:
    stop = start + count
    return np.array([f"seq{str(i).zfill(pad)}" for i in range(start, stop)], dtype=object)


def shard_path(outdir: Path, prefix: str, shard_idx: int, fmt: str, gz: bool) -> Path:
    ext = ".fasta" if fmt == "fasta" else ".csv"
    if gz:
        ext += ".gz"
    return outdir / f"{prefix}.part{str(shard_idx).zfill(4)}{ext}"


def generate_shard(
    rng: np.random.Generator,
    total_needed: int,
    length: int,
    alphabet: str,
    gc_min: Optional[float],
    gc_max: Optional[float],
    batch_size: int,
    allow_duplicates: bool,
    unique_scope: str,
) -> np.ndarray:
    """
    Generate up to `total_needed` sequences (as integer-encoded matrix),
    applying GC filters. Returns int_mat of shape (N, L) where N <= total_needed.
    """
    alpha = ALPHABETS[alphabet]
    collected = []
    seen = None
    if (not allow_duplicates) and unique_scope == "shard":
        seen = set()

    def collected_n():
        return sum(arr.shape[0] for arr in collected)

    while collected_n() < total_needed:
        ints = rng.integers(0, len(alpha), size=(batch_size, length), dtype=np.uint8)  # (B, L)

        # GC filter if needed
        if gc_min is not None or gc_max is not None:
            gc = compute_gc_frac(ints)
            mask = np.ones(gc.shape, dtype=bool)
            if gc_min is not None:
                mask &= (gc >= gc_min)
            if gc_max is not None:
                mask &= (gc <= gc_max)
            ints = ints[mask]
            if ints.size == 0:
                continue

        # De-dup within shard if requested (approximate by 64-bit hash of row bytes)
        if seen is not None:
            keep_rows = []
            for row in ints:
                h = hash(row.tobytes())
                if h not in seen:
                    seen.add(h)
                    keep_rows.append(row)
            if keep_rows:
                ints = np.stack(keep_rows, axis=0)
            else:
                ints = np.empty((0, length), dtype=np.uint8)

        if ints.shape[0] > 0:
            need = total_needed - collected_n()
            if ints.shape[0] > need:
                ints = ints[:need]
            collected.append(ints)

    return np.concatenate(collected, axis=0)


def main():
    args = parse_args()
    if args.num <= 0:
        raise ValueError("--num must be > 0")
    if args.length <= 0:
        raise ValueError("--length must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    outdir = ensure_outdir(args.outdir)

    rng = np.random.default_rng(args.seed)
    total = int(args.num)
    L = int(args.length)
    chunk = int(args.chunk_size)
    batch = args.batch_size or min(10_000_000, max(1, 3 * chunk))

    if batch < chunk:
        print(f"[warn] batch-size ({batch}) < chunk-size ({chunk}); generation may be slower.", file=sys.stderr)

    n_shards = math.ceil(total / chunk)
    print(f"[info] Target sequences: {total:,} | length: {L} | shards: {n_shards} | shard_size: {chunk:,}")
    print(f"[info] Batch-size: {batch:,} | format: {args.format} | gzip: {args.gzip} | alphabet: {args.alphabet}")
    if args.gc_min is not None or args.gc_max is not None:
        print(f"[info] GC filter: [{args.gc_min if args.gc_min is not None else '-'}, "
              f"{args.gc_max if args.gc_max is not None else '-'}]")

    next_id = args.start_index
    for shard_idx in range(1, n_shards + 1):
        shard_need = min(chunk, total - (shard_idx - 1) * chunk)
        print(f"[info] Generating shard {shard_idx}/{n_shards} -> need {shard_need:,} seqs...", file=sys.stderr)

        ints = generate_shard(
            rng=rng,
            total_needed=shard_need,
            length=L,
            alphabet=args.alphabet,
            gc_min=args.gc_min,
            gc_max=args.gc_max,
            batch_size=batch,
            allow_duplicates=args.allow_duplicates,
            unique_scope=args.unique_scope,
        )

        seqs = ints_to_strs(ints, ALPHABETS[args.alphabet])
        ids = make_ids(next_id, seqs.shape[0], args.pad_width)
        next_id += seqs.shape[0]

        out_path = shard_path(outdir, args.prefix, shard_idx, args.format, args.gzip)
        if args.format == "fasta":
            write_fasta_shard(out_path, ids, seqs, args.gzip)
        else:
            write_csv_shard(out_path, ids, seqs, args.gzip)

        print(f"[done] Wrote {seqs.shape[0]:,} sequences -> {out_path}", file=sys.stderr)

    print("[ok] All shards complete.")


if __name__ == "__main__":
    main()
