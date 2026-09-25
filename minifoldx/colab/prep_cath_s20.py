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
parser.add_argument("--workers",   type=int, default=8)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = Path(args.out).expanduser() if args.out else DATA_DIR / "cath_s20_coords.npz"

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


# ── Install gemmi if needed ───────────────────────────────────────────────────
try:
    import gemmi
except ImportError:
    log("Installing gemmi ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gemmi"], check=True)
    import gemmi


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


# ── Build PDB path index ──────────────────────────────────────────────────────
log("=== Indexing PDB files ===")
pdb_index = {}
for p in PDB_DIR.rglob("*"):
    if p.is_file():
        pdb_index[p.stem] = p   # stem strips .pdb/.ent; CATH files have no ext so stem==name
log(f"  {len(pdb_index)} files indexed")


# ── Extract Cβ coords with gemmi ──────────────────────────────────────────────
log(f"=== Extracting Cβ coordinates (len {args.min_len}–{args.max_len}, {args.workers} workers) ===")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def process(item):
    domain_id, seq = item
    L = len(seq)
    if L < args.min_len or L > args.max_len:
        return None
    pdb_path = pdb_index.get(domain_id)
    if pdb_path is None:
        return None
    try:
        st    = gemmi.read_pdb(str(pdb_path))
        model = st[0]
        res_coords = []
        for chain in model:
            for res in chain:
                info = gemmi.find_tabulated_residue(res.name)
                aa = info.one_letter_code if info.found() else "X"
                if aa not in VALID_AA:
                    continue
                atom = res.find_atom("CB", "\0") or res.find_atom("CA", "\0")
                if atom is None:
                    continue
                res_coords.append((aa, atom.pos.x, atom.pos.y, atom.pos.z))
        if len(res_coords) < args.min_len:
            return None
        pdb_seq = "".join(r[0] for r in res_coords)
        coords  = np.array([[r[1], r[2], r[3]] for r in res_coords], dtype=np.float32)
        return domain_id, pdb_seq, coords
    except Exception as e:
        return ("ERROR", str(e), None)


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

items = [(did, seq) for did, seq in seqs.items()
         if args.min_len <= len(seq) <= args.max_len and did in pdb_index]
log(f"  {len(items)} candidates to process")

domains_out = {}
done = 0
with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
    futs = {ex.submit(process, it): it for it in items}
    for fut in as_completed(futs):
        done += 1
        res = fut.result()
        if res and res[0] == "ERROR" and done < 5:
            log(f"  ERROR sample: {res[1]}")
        elif res and res[0] != "ERROR":
            domains_out[res[0]] = {"seq": res[1], "coords": res[2]}
        if done % 1000 == 0:
            log(f"  {done}/{len(items)}  found {len(domains_out)}")

log(f"  {len(domains_out)} domains with coordinates")


# ── Save .npz ─────────────────────────────────────────────────────────────────
log(f"=== Saving {OUT_PATH.name} ===")
domain_ids  = list(domains_out.keys())
sequences   = [domains_out[d]["seq"]    for d in domain_ids]
coords_list = [domains_out[d]["coords"] for d in domain_ids]
lengths     = np.array([len(c) for c in coords_list], dtype=np.int32)
coords_flat = np.concatenate(coords_list, axis=0)

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
    )
    if result.returncode == 0:
        log("  Upload complete.")
    else:
        log("  Upload failed — check rclone config.")
else:
    log(f"Skipped upload. To upload manually:\n  rclone copyto {OUT_PATH} {DRIVE_DEST}")

log(f"Done. Total time: {(time.time()-T0)/60:.1f} min")
