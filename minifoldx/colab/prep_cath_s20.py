#!/usr/bin/env python3
"""
Download CATH S20, extract Cβ coordinates, save as cath_s20_coords.npz,
and upload to Google Drive so Colab training runs skip the slow extraction step.

Usage:
    python prep_cath_s20.py
    python prep_cath_s20.py --data-dir ~/data/cath_s20 --min-len 40 --max-len 300

Uploads to: gdrive:Colab_Data/minifold_colab/weights/cath_s20_coords.npz
"""

import argparse
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="~/Science/Programs/MiniFoldX/data/cath_s20")
parser.add_argument("--out",      default=None, help="Output .npz path (default: data-dir/cath_s20_coords.npz)")
parser.add_argument("--min-len",  type=int, default=40)
parser.add_argument("--max-len",  type=int, default=300)
parser.add_argument("--no-upload", action="store_true", help="Skip rclone upload")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = Path(args.out).expanduser() if args.out else DATA_DIR / "cath_s20_coords.npz"

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


# ── Download ──────────────────────────────────────────────────────────────────
CATH_BASE = "http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets"
FA_PATH  = DATA_DIR / "cath_s20.fa"
PDB_TGZ  = DATA_DIR / "cath_s20.pdb.tgz"
PDB_DIR  = DATA_DIR / "pdbs"

log("=== Downloading CATH S20 ===")
if not FA_PATH.exists():
    log("  Downloading FASTA ...")
    urllib.request.urlretrieve(f"{CATH_BASE}/cath-dataset-nonredundant-S20.fa", str(FA_PATH))
    log(f"  Saved {FA_PATH.name}")
else:
    log(f"  {FA_PATH.name} already cached")

if not PDB_TGZ.exists():
    log("  Downloading PDB structures (~350 MB) ...")
    urllib.request.urlretrieve(f"{CATH_BASE}/cath-dataset-nonredundant-S20.pdb.tgz", str(PDB_TGZ))
    log(f"  Saved {PDB_TGZ.name}")
else:
    log(f"  {PDB_TGZ.name} already cached")

if not PDB_DIR.exists():
    log("  Extracting PDB files ...")
    PDB_DIR.mkdir()
    with tarfile.open(PDB_TGZ, "r:gz") as tf:
        tf.extractall(PDB_DIR, filter="data")
    log(f"  Extracted to {PDB_DIR}/")
else:
    log(f"  PDB dir already extracted")


# ── Parse FASTA ───────────────────────────────────────────────────────────────
log("=== Parsing FASTA ===")
seqs = {}
with open(FA_PATH) as fh:
    domain_id = None
    buf = []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if domain_id:
                seqs[domain_id] = "".join(buf)
            raw = line[1:].split()[0]
            parts = raw.split("|")
            domain_id = parts[2].split("/")[0] if len(parts) >= 3 else raw.split("/")[0]
            buf = []
        else:
            buf.append(line)
    if domain_id:
        seqs[domain_id] = "".join(buf)
log(f"  {len(seqs)} sequences")


# ── Extract Cβ coords ─────────────────────────────────────────────────────────
log(f"=== Extracting Cβ coordinates (len {args.min_len}–{args.max_len}) ===")

AA3TO1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
           "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
           "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def parse_cb(pdb_path: Path):
    residues = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom  = line[12:16].strip()
            res3  = line[17:20].strip()
            resid = int(line[22:26].strip())
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if res3 not in AA3TO1:
                continue
            if resid not in residues:
                residues[resid] = {"aa": AA3TO1[res3], "CA": None, "CB": None}
            if atom == "CA":
                residues[resid]["CA"] = xyz
            elif atom == "CB":
                residues[resid]["CB"] = xyz
    seq, coords = [], []
    for r in (residues[k] for k in sorted(residues)):
        c = r["CB"] if r["CB"] is not None else r["CA"]
        if c is None:
            continue
        seq.append(r["aa"])
        coords.append(c)
    return "".join(seq), np.stack(coords).astype(np.float32) if coords else None


from concurrent.futures import ThreadPoolExecutor, as_completed

def process(item):
    domain_id, seq = item
    L = len(seq)
    if L < args.min_len or L > args.max_len:
        return None
    hits = (list(PDB_DIR.glob(f"**/{domain_id}"))
          + list(PDB_DIR.glob(f"**/{domain_id}.pdb"))
          + list(PDB_DIR.glob(f"**/{domain_id}.ent")))
    if not hits:
        return None
    pdb_seq, coords = parse_cb(hits[0])
    if coords is None or len(coords) < args.min_len:
        return None
    return domain_id, pdb_seq, coords

domains_out = {}
items = list(seqs.items())
n = len(items)
done = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(process, it): it for it in items}
    for fut in as_completed(futs):
        done += 1
        res = fut.result()
        if res:
            domains_out[res[0]] = {"seq": res[1], "coords": res[2]}
        if done % 1000 == 0:
            log(f"  {done}/{n}  found {len(domains_out)} so far")

log(f"  {len(domains_out)} domains with coordinates")


# ── Save .npz ─────────────────────────────────────────────────────────────────
log(f"=== Saving {OUT_PATH.name} ===")
domain_ids = list(domains_out.keys())
sequences  = [domains_out[d]["seq"]    for d in domain_ids]
coords_list= [domains_out[d]["coords"] for d in domain_ids]
lengths    = np.array([len(c) for c in coords_list], dtype=np.int32)
coords_flat= np.concatenate(coords_list, axis=0)           # (sum_L, 3) float32

np.savez_compressed(
    str(OUT_PATH),
    domain_ids  = np.array(domain_ids),
    sequences   = np.array(sequences, dtype=object),
    coords_flat = coords_flat,
    lengths     = lengths,
    min_len     = np.int32(args.min_len),
    max_len     = np.int32(args.max_len),
)
size_mb = OUT_PATH.stat().st_size / 1e6
log(f"  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
log(f"  {len(domain_ids)} domains  |  coords_flat shape: {coords_flat.shape}")


# ── Upload to Drive ───────────────────────────────────────────────────────────
DRIVE_DEST = "gdrive:Colab_Data/minifold_colab/weights/cath_s20_coords.npz"

if not args.no_upload:
    log(f"=== Uploading to Drive ===")
    log(f"  → {DRIVE_DEST}")
    result = subprocess.run(
        ["rclone", "copyto", str(OUT_PATH), DRIVE_DEST, "--progress"],
        capture_output=False,
    )
    if result.returncode == 0:
        log("  Upload complete.")
    else:
        log("  Upload failed — check rclone config.")
else:
    log(f"Skipped upload. To upload manually:\n  rclone copyto {OUT_PATH} {DRIVE_DEST}")

log(f"Done. Total time: {(time.time()-T0)/60:.1f} min")
