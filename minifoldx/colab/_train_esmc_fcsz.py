#!/usr/bin/env python3
"""
Train fc_s and fc_z for ESMC 600M → MiniFoldX 12L trunk.

Designed to run on Google Colab A100. Self-contained: installs deps,
downloads CATH S20 data and MiniFoldX weights, trains, saves checkpoints.

Usage (Colab cell):
    !python train_esmc_fcsz.py
    !python train_esmc_fcsz.py --epochs 20 --lr 5e-4 --out /content/drive/MyDrive/minifold_esmc

Architecture:
    ESMC 600M (frozen, fp16)
        hidden states  (L, 1152)  → fc_s (trainable) → (L, 1024)
        attention wts  (L, L, 648)  → fc_z (trainable) → (L, L, 128)
    MiniFoldX 12L trunk (frozen, fp16)
        → distogram logits (L, L, 64)

Loss: cross-entropy vs true Cβ–Cβ distance bins from CATH S20 PDB files.
Train/test split: random 80/20 (S20 guarantees <20% identity within set,
so no clustering step needed).
"""

import argparse
import os
import random
import sys
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

# ── Argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",    type=int,   default=15)
parser.add_argument("--lr",        type=float, default=1e-3)
parser.add_argument("--max-len",   type=int,   default=300,   help="Skip domains longer than this")
parser.add_argument("--min-len",   type=int,   default=40,    help="Skip domains shorter than this")
parser.add_argument("--seed",      type=int,   default=42)
parser.add_argument("--out",       type=str,   default="/content/outputs")
parser.add_argument("--weights",   type=str,   default="/content/weights")
parser.add_argument("--data",      type=str,   default="/content/cath_s20")
parser.add_argument("--no-drive",  action="store_true", help="Skip Drive checkpointing")
parser.add_argument("--ckpt-every",type=int,   default=500,   help="Checkpoint every N steps")
args = parser.parse_args()

OUT_DIR     = Path(args.out);     OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR = Path(args.weights); WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = Path(args.data);    DATA_DIR.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


# ── Step helpers ─────────────────────────────────────────────────────────────
def run(*cmd):
    log("+ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))


# ── 1. Install dependencies ──────────────────────────────────────────────────
log("=== Installing dependencies ===")
run(sys.executable, "-m", "pip", "install", "-q", "uv")
run("uv", "pip", "install", "--system", "-q",
    "git+https://github.com/ZacharyArdern/MiniFoldX.git#subdirectory=minifoldx/pytorch",
    "git+https://github.com/evolutionaryScale/esm.git",
    "safetensors", "huggingface_hub[hf_xet]", "einops",
    "dm-tree", "ml-collections", "modelcif", "edit_distance", "fair-esm")


# ── 2. Download CATH S20 (or load pre-processed .npz from weights cache) ─────
COORDS_NPZ = WEIGHTS_DIR / "cath_s20_coords.npz"
CATH_BASE  = "http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets"

if not COORDS_NPZ.exists():
    log("=== Downloading CATH S20 ===")
    FA_PATH = DATA_DIR / "cath_s20.fa"
    PDB_TGZ = DATA_DIR / "cath_s20.pdb.tgz"
    PDB_DIR = DATA_DIR / "pdbs"

    if not FA_PATH.exists():
        urllib.request.urlretrieve(f"{CATH_BASE}/cath-dataset-nonredundant-S20.fa", str(FA_PATH))
        log(f"  Downloaded {FA_PATH.name}")
    if not PDB_TGZ.exists():
        log("  Downloading PDB structures (~350 MB) ...")
        urllib.request.urlretrieve(f"{CATH_BASE}/cath-dataset-nonredundant-S20.pdb.tgz", str(PDB_TGZ))
        log(f"  Downloaded {PDB_TGZ.name}")
    if not PDB_DIR.exists():
        log("  Extracting PDB files ...")
        PDB_DIR.mkdir()
        with tarfile.open(PDB_TGZ, "r:gz") as tf:
            tf.extractall(PDB_DIR)
        log(f"  Extracted to {PDB_DIR}/")
else:
    log(f"=== Using pre-processed coords cache: {COORDS_NPZ.name} ===")


# ── 3. Download MiniFoldX 12L checkpoint ─────────────────────────────────────
log("=== Downloading MiniFoldX 12L weights ===")
CKPT_PATH = WEIGHTS_DIR / "minifold_12L.safetensors"
if not CKPT_PATH.exists():
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
    hf_hub_download(repo_id="z-ardern/MiniFoldX_weights",
                    filename="minifold_12L.safetensors",
                    local_dir=str(WEIGHTS_DIR), local_dir_use_symlinks=False)
    log(f"  Saved to {CKPT_PATH}")
else:
    log(f"  Using cached {CKPT_PATH.name}")


# ── 4 & 5. Load domains (from .npz cache or raw PDB files) ───────────────────
import numpy as np

domains = {}

if COORDS_NPZ.exists():
    log("=== Loading domains from .npz cache ===")
    data = np.load(str(COORDS_NPZ), allow_pickle=True)
    ids      = data["domain_ids"].tolist()
    seqs_arr = data["sequences"].tolist()
    lengths  = data["lengths"]
    coords_f = data["coords_flat"]
    offset   = 0
    for did, seq, L in zip(ids, seqs_arr, lengths):
        L = int(L)
        if args.min_len <= len(seq) <= args.max_len:
            domains[did] = {"seq": seq, "coords": coords_f[offset:offset+L]}
        offset += L
    log(f"  {len(domains)} domains in length range [{args.min_len},{args.max_len}]")

else:
    log("=== Parsing CATH S20 sequences ===")
    seqs = {}
    with open(FA_PATH) as fh:
        domain_id = None
        seq_buf = []
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if domain_id:
                    seqs[domain_id] = "".join(seq_buf)
                raw_id = line[1:].split()[0]
                parts  = raw_id.split("|")
                domain_id = parts[2].split("/")[0] if len(parts) >= 3 else raw_id.split("/")[0]
                seq_buf = []
            else:
                seq_buf.append(line)
        if domain_id:
            seqs[domain_id] = "".join(seq_buf)
    log(f"  {len(seqs)} sequences parsed")

    log("=== Extracting Cβ coordinates ===")
    aa3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
               "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
               "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

    def parse_cb_coords(pdb_path: Path):
        residues = {}
        with open(pdb_path) as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom_name = line[12:16].strip()
                res_name  = line[17:20].strip()
                res_seq   = int(line[22:26].strip())
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                if res_name not in aa3to1:
                    continue
                if res_seq not in residues:
                    residues[res_seq] = {"aa": aa3to1[res_name], "CA": None, "CB": None}
                if atom_name == "CA":
                    residues[res_seq]["CA"] = np.array([x, y, z])
                elif atom_name == "CB":
                    residues[res_seq]["CB"] = np.array([x, y, z])
        seq, coords = [], []
        for r in (residues[k] for k in sorted(residues)):
            c = r["CB"] if r["CB"] is not None else r["CA"]
            if c is None:
                continue
            seq.append(r["aa"])
            coords.append(c)
        return "".join(seq), np.stack(coords) if coords else None

    for domain_id, seq in seqs.items():
        L = len(seq)
        if L < args.min_len or L > args.max_len:
            continue
        candidates = (list(PDB_DIR.glob(f"**/{domain_id}"))
                    + list(PDB_DIR.glob(f"**/{domain_id}.pdb"))
                    + list(PDB_DIR.glob(f"**/{domain_id}.ent")))
        if not candidates:
            continue
        pdb_seq, coords = parse_cb_coords(candidates[0])
        if coords is None or len(coords) < args.min_len:
            continue
        domains[domain_id] = {"seq": pdb_seq, "coords": coords}

    log(f"  {len(domains)} domains with coordinates in length range [{args.min_len},{args.max_len}]")


# ── 6. Distance binning ────────────────────────────────────────────────────────
# 64 bins from 2.3125 to 21.6875 Å (ESMFold convention)
N_BINS  = 64
D_MIN   = 2.3125
D_MAX   = 21.6875
BIN_EDGES = np.linspace(D_MIN, D_MAX, N_BINS + 1)

def dist_to_bins(coords: np.ndarray) -> np.ndarray:
    """(L,3) coords → (L,L) int64 bin indices."""
    diff  = coords[:, None] - coords[None, :]      # (L,L,3)
    dists = np.sqrt((diff**2).sum(-1))              # (L,L)
    bins  = np.digitize(dists, BIN_EDGES[1:-1])    # 0..63
    return bins.astype(np.int64)


# ── 7. Train/test split ────────────────────────────────────────────────────────
all_ids = sorted(domains.keys())
random.shuffle(all_ids)
n_train = int(0.8 * len(all_ids))
train_ids = all_ids[:n_train]
test_ids  = all_ids[n_train:]
log(f"  Split: {len(train_ids)} train / {len(test_ids)} test")


# ── 8. Load models ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

log("=== Loading ESMC 600M ===")
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
esmc = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M")
esmc = esmc.half().cuda().eval()
for p in esmc.parameters():
    p.requires_grad_(False)
esmc_tok = EsmcTokenizer()
log("  ESMC 600M ready")

log("=== Loading MiniFoldX 12L trunk ===")
from minifold.model.miniformer import MiniFormer as MiniFormerPT

class SequenceToPair(nn.Module):
    def __init__(self, c_s, inner_dim, c_z):
        super().__init__()
        self.layernorm = nn.LayerNorm(c_s)   # matches checkpoint key 'seq_to_pair.layernorm'
        self.proj   = nn.Linear(c_s, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(inner_dim * 2, c_z, bias=True)
    def forward(self, s):
        s = self.proj(self.layernorm(s))
        q, k = s.chunk(2, dim=-1)
        return self.o_proj(torch.cat([q[:,None,:,:]*k[:,:,None,:],
                                       q[:,None,:,:]-k[:,:,None,:]], dim=-1))

class RelativePosition(nn.Module):
    def __init__(self, bins, c_z):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2*bins+2, c_z)
    def forward(self, residue_index, mask):
        diff = (residue_index[:,None,:] - residue_index[:,:,None]).clamp(-self.bins, self.bins) + self.bins + 1
        diff[mask == 0] = 0
        return self.embedding(diff)

class FoldingTrunk(nn.Module):
    def __init__(self, c_s=1024, c_z=128, bins=32, disto_bins=64, num_layers=12):
        super().__init__()
        self.disto_bins = disto_bins
        self.positional_embedding = RelativePosition(bins, c_z)
        self.seq_to_pair = SequenceToPair(c_s, c_z//2, c_z)
        self.projection  = nn.Linear(c_z*3, c_z)
        self.recycle     = nn.Linear(disto_bins, c_z)
        self.miniformer  = MiniFormerPT(c_z, blocks=num_layers, kernels=False)
        self.fc_out      = nn.Sequential(nn.Linear(c_z, c_z), nn.ReLU(), nn.Linear(c_z, disto_bins))
    def forward(self, s_s, s_z, mask, num_recycling=0):
        pair_mask = (mask[:,None,:]*mask[:,:,None]).to(s_z)
        residx = torch.arange(s_s.shape[1], device=s_s.device).unsqueeze(0).expand(s_s.shape[0], -1)
        s_z = self.projection(torch.cat([s_z, self.seq_to_pair(s_s),
                                          self.positional_embedding(residx, mask.bool())], dim=-1))
        dists = torch.zeros(*s_z.shape[:3], self.disto_bins, device=s_z.device, dtype=s_z.dtype)
        for _ in range(num_recycling+1):
            s_z_c = self.miniformer(s_z + self.recycle(dists), pair_mask)
            preds = self.fc_out(s_z_c + s_z_c.transpose(1,2))
            dists = F.one_hot(preds.detach().argmax(-1), self.disto_bins).to(s_z)
        return preds, s_z_c

# Load trunk weights from checkpoint
sd = load_file(str(CKPT_PATH), device="cpu")
trunk = FoldingTrunk(c_s=1024, c_z=128, bins=32, disto_bins=64, num_layers=12)
trunk_sd = {}
for k, v in sd.items():
    if not k.startswith("model.fold."):
        continue
    new_k = k.replace("model.fold.", "")
    new_k = new_k.replace("miniformer._orig_mod.", "miniformer.")
    trunk_sd[new_k] = v
missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
if missing:
    log(f"  WARNING: {len(missing)} missing trunk keys: {missing[:3]}")
trunk = trunk.half().cuda().eval()
for p in trunk.parameters():
    p.requires_grad_(False)
log(f"  Trunk loaded ({len(trunk_sd)} tensors matched)")


# ── 9. Trainable projections ───────────────────────────────────────────────────
log("=== Building trainable fc_s and fc_z ===")
C_S, C_Z = 1024, 128
ESMC_HIDDEN = 1152   # ESMC 600M hidden dim
ESMC_ATTN   = 648    # 36 layers × 18 heads

fc_s = nn.Sequential(nn.Linear(ESMC_HIDDEN, C_S), nn.ReLU(), nn.Linear(C_S, C_S)).cuda()  # fp32
fc_z = nn.Sequential(nn.Linear(ESMC_ATTN,  C_Z), nn.ReLU(), nn.Linear(C_Z, C_Z)).cuda()   # fp32
log(f"  fc_s params: {sum(p.numel() for p in fc_s.parameters()):,}")
log(f"  fc_z params: {sum(p.numel() for p in fc_z.parameters()):,}")

optimizer = torch.optim.AdamW(
    list(fc_s.parameters()) + list(fc_z.parameters()),
    lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs * len(train_ids))


# ── 10. Forward + loss helpers ────────────────────────────────────────────────
def esmc_forward(seq: str):
    """Run ESMC, return (hidden, attentions) with special tokens stripped."""
    encoded = esmc_tok([seq], return_tensors="pt", padding=False)
    input_ids = encoded["input_ids"].cuda()
    with torch.no_grad():
        out = esmc(input_ids=input_ids, output_attentions=True)
    hidden = out.last_hidden_state[:, 1:-1].float()    # (1, L, 1152) fp32, strip BOS/EOS
    # attentions: tuple of 36 × (1, 18, L+2, L+2)
    attns = torch.stack(out.attentions, dim=1)         # (1, 36, 18, L+2, L+2)
    attns = attns[:, :, :, 1:-1, 1:-1].float()         # strip BOS/EOS → (1, 36, 18, L, L) fp32
    L = hidden.shape[1]
    s_z_in = attns.permute(0,3,4,1,2).reshape(1, L, L, ESMC_ATTN)  # (1,L,L,648)
    return hidden, s_z_in


def distogram_loss(preds: torch.Tensor, true_bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    preds:     (1, L, L, 64)  — trunk logits
    true_bins: (L, L)         — int64 target bin indices
    mask:      (1, L)         — padding mask
    Returns scalar CE loss over valid (i,j) pairs.
    """
    L = preds.shape[1]
    pair_mask = (mask[:,None,:] * mask[:,:,None]).squeeze(0)  # (L,L)
    logits = preds.squeeze(0)                                  # (L,L,64)
    # Symmetrise: average (i,j) and (j,i) predictions
    logits_sym = (logits + logits.transpose(0,1)) / 2
    loss = F.cross_entropy(logits_sym.reshape(-1, N_BINS),
                           true_bins.reshape(-1).cuda(),
                           reduction="none")
    loss = (loss * pair_mask.reshape(-1)).sum() / pair_mask.sum().clamp(min=1)
    return loss


# ── 11. Training loop ─────────────────────────────────────────────────────────
log("=== Training ===")
log(f"  Epochs: {args.epochs}  LR: {args.lr}  Domains: {len(train_ids)} train / {len(test_ids)} test")

best_test_loss = float("inf")
global_step    = 0

for epoch in range(args.epochs):
    fc_s.train(); fc_z.train()
    random.shuffle(train_ids)
    epoch_losses = []

    for domain_id in train_ids:
        d   = domains[domain_id]
        seq = d["seq"]
        true_bins = torch.from_numpy(dist_to_bins(d["coords"]))  # (L,L)

        try:
            hidden, s_z_in = esmc_forward(seq)
        except Exception as e:
            log(f"  WARN: ESMC failed on {domain_id}: {e}")
            continue

        L = hidden.shape[1]
        if L != len(seq):
            # sequence length mismatch after stripping special tokens — skip
            continue

        mask = torch.ones(1, L, dtype=torch.float16, device="cuda")

        # Project in fp32, then cast to fp16 for frozen trunk
        s_s = fc_s(hidden).half()            # (1,L,1024)
        s_z = fc_z(s_z_in).half()            # (1,L,L,128)

        # Trunk forward (frozen weights, but in compute graph for gradients)
        preds, _ = trunk(s_s, s_z, mask, num_recycling=0)  # (1,L,L,64)

        loss = distogram_loss(preds, true_bins, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(fc_s.parameters()) + list(fc_z.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())
        global_step += 1

        if global_step % 100 == 0:
            log(f"  step {global_step}  loss {sum(epoch_losses[-50:])/min(50,len(epoch_losses[-50:])):.4f}"
                f"  lr {scheduler.get_last_lr()[0]:.2e}")

        if global_step % args.ckpt_every == 0:
            ckpt = OUT_DIR / f"fcsz_step{global_step}.pt"
            torch.save({"fc_s": fc_s.state_dict(), "fc_z": fc_z.state_dict(),
                        "step": global_step, "epoch": epoch}, str(ckpt))
            log(f"  Checkpoint saved: {ckpt.name}")

    # Validation
    fc_s.eval(); fc_z.eval()
    test_losses = []
    with torch.no_grad():
        for domain_id in test_ids[:200]:  # sample 200 for speed
            d   = domains[domain_id]
            seq = d["seq"]
            true_bins = torch.from_numpy(dist_to_bins(d["coords"]))
            try:
                hidden, s_z_in = esmc_forward(seq)
            except Exception:
                continue
            L = hidden.shape[1]
            if L != len(seq):
                continue
            mask = torch.ones(1, L, dtype=torch.float16, device="cuda")
            s_s = fc_s(hidden).half(); s_z = fc_z(s_z_in).half()
            preds, _ = trunk(s_s, s_z, mask, num_recycling=0)
            test_losses.append(distogram_loss(preds, true_bins, mask).item())

    train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
    test_loss  = sum(test_losses)  / max(len(test_losses),  1)
    log(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        ckpt = OUT_DIR / "fcsz_best.pt"
        torch.save({"fc_s": fc_s.state_dict(), "fc_z": fc_z.state_dict(),
                    "epoch": epoch, "test_loss": test_loss}, str(ckpt))
        log(f"  Best model saved (test_loss={test_loss:.4f})")


# ── 12. Final save ────────────────────────────────────────────────────────────
log("=== Saving final checkpoint ===")
final_ckpt = OUT_DIR / "fcsz_final.pt"
torch.save({"fc_s": fc_s.state_dict(), "fc_z": fc_z.state_dict(),
            "epochs": args.epochs, "best_test_loss": best_test_loss,
            "esmc_model": "biohub/ESMC-600M",
            "trunk": "minifold_12L", "bins": N_BINS,
            "d_min": D_MIN, "d_max": D_MAX}, str(final_ckpt))
log(f"  Saved: {final_ckpt}")
log(f"  Best test loss: {best_test_loss:.4f}")
log(f"  Total time: {(time.time()-T0)/60:.1f} min")
