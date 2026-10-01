#!/usr/bin/env python3
"""
Colab: train fc_s + fc_z + bb_update + pLDDT head with Ca distance loss.

ESM2-650M stays loaded on GPU throughout. Embeddings are buffered in GPU memory
(BUFFER_SIZE sequences at a time), with STEPS_PER_BUFFER gradient steps per buffer
before refreshing. No disk caching.

Inputs:  /content/weights/cath_s20_coords.npz
Outputs: /content/outputs/fcsz_fape_650m.pt
         /content/outputs/fcsz_fape_650m_step<N>.pt  (every SAVE_EVERY steps)
         /content/outputs/fcsz_fape_650m_resume.pt
"""

import os, random, math, time, subprocess, sys, types
from pathlib import Path

# ── 1. Install dependencies ────────────────────────────────────────────────────
def run(*cmd):
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))

print("=== Installing dependencies ===")
run(sys.executable, "-m", "pip", "install", "-q", "uv")
run("uv", "pip", "install", "--system", "-q",
    "git+https://github.com/ZacharyArdern/MiniFoldX.git#subdirectory=minifoldx/pytorch",
    "transformers", "safetensors", "huggingface_hub[hf_xet]", "einops",
    "dm-tree", "ml-collections", "modelcif", "edit_distance", "fair-esm")

# ── 2. Paths + config ─────────────────────────────────────────────────────────
WEIGHTS_DIR = Path("/content/weights"); WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("/content/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_PATH    = WEIGHTS_DIR / "cath_s20_coords.npz"
CKPT_PATH   = WEIGHTS_DIR / "minifold_12L.safetensors"
OUT_PATH    = OUT_DIR / "fcsz_fape_650m.pt"
RESUME_PATH = OUT_DIR / "fcsz_fape_650m_resume.pt"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
N_SEQS          = 8000   # sequences sampled from CATH S20
N_STEPS         = 30000
LR              = 3e-4
WARMUP          = 1000
SEED            = 42
SAVE_EVERY      = 3000
RECYCLING       = 1
BUFFER_SIZE     = 1000   # sequences held in GPU memory at once
STEPS_PER_BUFFER = 80    # gradient steps per buffer before refreshing

random.seed(SEED); torch.manual_seed(SEED)
T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

if not NPZ_PATH.exists():
    raise FileNotFoundError(f"{NPZ_PATH} not found — upload it to /content/weights/ first")

# ── 3. Load CATH S20 ──────────────────────────────────────────────────────────
log("Loading CATH S20 ...")
data    = np.load(str(NPZ_PATH), allow_pickle=True)
all_ids = data["domain_ids"].tolist()
all_seqs = data["sequences"].tolist()
lengths  = data["lengths"]
offsets  = np.concatenate([[0], np.cumsum(lengths)])
cf       = data["coords_flat"]

ca_map     = {did: torch.tensor(cf[offsets[i]:offsets[i+1]], dtype=torch.float32)
              for i, did in enumerate(all_ids)}
seq_lookup = dict(zip(all_ids, all_seqs))

random.seed(SEED)
indices  = random.sample(range(len(all_ids)), N_SEQS)
did_list = [all_ids[i] for i in indices]
seq_list = [all_seqs[i] for i in indices]
log(f"  {len(all_ids)} total  sampled {N_SEQS}  mean_len={sum(len(s) for s in seq_list)//N_SEQS}")

# ── 4. ESM v3 mock (needed by MiniFoldX imports) ──────────────────────────────
import esm as _esm_mod
_dm = types.ModuleType("esm.data")
class _FA:
    @staticmethod
    def from_architecture(n): return _FA()
_dm.Alphabet = _FA
_esm_mod.data = _dm
sys.modules["esm.data"] = _dm

# ── 5. Download MiniFoldX checkpoint ──────────────────────────────────────────
if not CKPT_PATH.exists():
    log("Downloading MiniFoldX 12L weights ...")
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
    hf_hub_download(repo_id="z-ardern/MiniFoldX_weights",
                    filename="minifold_12L.safetensors",
                    local_dir=str(WEIGHTS_DIR), local_dir_use_symlinks=False)

# ── 6. Load ESM2-650M (stays on GPU throughout) ───────────────────────────────
from transformers import EsmModel, EsmTokenizer
log("Loading ESM2-650M ...")
tok = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                attn_implementation="eager").half().to(DEVICE).eval()
for p in esm.parameters(): p.requires_grad_(False)
log("  ESM2-650M ready")

# ── 7. Load frozen trunk + SM ─────────────────────────────────────────────────
log("Loading MiniFoldX trunk + SM ...")
from safetensors.torch import load_file
from minifold.model.model import FoldingTrunk, PairToSequence
from minifold.model.structure import StructureModule
from minifold.model.heads import PerResidueLDDTCaPredictor
from minifold.data.config import model_config
from minifold.data.of_data import of_inference

sd = load_file(str(CKPT_PATH), device="cpu")

trunk = FoldingTrunk(c_s=1024, c_z=128, bins=32, disto_bins=64, num_layers=12, kernels=False)
trunk.load_state_dict({k.replace("model.fold.","").replace("miniformer._orig_mod.","miniformer."): v
                       for k,v in sd.items() if k.startswith("model.fold.")}, strict=False)
trunk = trunk.to(DEVICE).eval()
for p in trunk.parameters(): p.requires_grad_(False)

sz_proj = PairToSequence(c_z=128, c_s=1024, c_s_out=1024)
sz_proj.load_state_dict({k.replace("model.sz_project.",""): v
                         for k,v in sd.items() if "sz_project" in k})
sz_proj = sz_proj.to(DEVICE).eval()
for p in sz_proj.parameters(): p.requires_grad_(False)

sm = StructureModule(c_s=1024, c_z=128, c_resnet=128, head_dim=64, no_heads=16,
                     no_blocks=8, no_resnet_blocks=2, no_angles=7,
                     trans_scale_factor=10, epsilon=1e-5, inf=1e5)
sm.load_state_dict({k.replace("model.structure_module.",""): v
                    for k,v in sd.items() if "structure_module" in k})
sm = sm.to(DEVICE).eval()
for p in sm.parameters(): p.requires_grad_(False)
for p in sm.bb_update.parameters(): p.requires_grad_(True)

plddt_head = PerResidueLDDTCaPredictor(no_bins=50, c_in=1024, c_hidden=128)
plddt_head.load_state_dict({k.replace("model.aux_heads.plddt.", ""): v
                             for k, v in sd.items() if "aux_heads.plddt" in k})
plddt_head = plddt_head.to(DEVICE)
for p in plddt_head.parameters(): p.requires_grad_(True)

cfg = model_config("initial_training", train=False, low_prec=False, long_sequence_inference=False).data

def get_aatype(seq):
    f = of_inference(seq, "predict", cfg)
    return f["aatype"][:,0].to(DEVICE), f["seq_mask"][:,0].float().to(DEVICE)

log("  Trunk + SM ready")

# ── 8. Trainable fc_s + fc_z ──────────────────────────────────────────────────
fc_s = nn.Sequential(nn.Linear(1280, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(DEVICE)
fc_z = nn.Sequential(nn.Linear(660,  128),  nn.ReLU(), nn.Linear(128,  128)).to(DEVICE)

start_step = 0
if RESUME_PATH.exists():
    log(f"Resuming from {RESUME_PATH.name} ...")
    resume = torch.load(str(RESUME_PATH), map_location="cpu")
    fc_s.load_state_dict(resume["fc_s"])
    fc_z.load_state_dict(resume["fc_z"])
    if "bb_update" in resume:
        sm.bb_update.load_state_dict(resume["bb_update"])
    if "plddt_head" in resume:
        plddt_head.load_state_dict(resume["plddt_head"])
    start_step = resume.get("step", 0)
    log(f"  Resuming from step {start_step}")

trainable_params = (list(fc_s.parameters()) + list(fc_z.parameters()) +
                    list(sm.bb_update.parameters()) + list(plddt_head.parameters()))
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, N_STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
for _ in range(start_step):
    scheduler.step()

# ── 9. Loss helpers ───────────────────────────────────────────────────────────
def ca_dist_loss(pred_ca, true_ca, clamp_ang=20.0):
    def pw(x):
        d = x.unsqueeze(0) - x.unsqueeze(1)
        return (d**2).sum(-1).clamp(min=1e-6).sqrt()
    return F.mse_loss(pw(pred_ca).clamp(max=clamp_ang),
                      pw(true_ca).clamp(max=clamp_ang))

def compute_lddt_ca(pred_ca, true_ca, cutoff=15.0, n_bins=50):
    with torch.no_grad():
        pred_d = torch.cdist(pred_ca.unsqueeze(0), pred_ca.unsqueeze(0))[0]
        true_d = torch.cdist(true_ca.unsqueeze(0), true_ca.unsqueeze(0))[0]
        mask   = (true_d < cutoff) & ~torch.eye(pred_ca.shape[0], dtype=torch.bool, device=pred_ca.device)
        diff   = (pred_d - true_d).abs()
        preserved = sum((diff < t).float() * mask for t in [0.5, 1.0, 2.0, 4.0])
        n_contacts = mask.float().sum(dim=1).clamp(min=1e-6)
        lddt   = preserved.sum(dim=1) / (4.0 * n_contacts)
        return (lddt * n_bins).long().clamp(0, n_bins - 1)

def lddt_ce_loss(plddt_logits, pred_ca, true_ca):
    bin_idx = compute_lddt_ca(pred_ca.detach(), true_ca)
    logits  = plddt_logits[0] if plddt_logits.dim() == 3 else plddt_logits
    return F.cross_entropy(logits, bin_idx)

PLDDT_WEIGHT = 0.01

# ── 10. Buffer fill ───────────────────────────────────────────────────────────
def fill_buffer(seq_slice):
    """Run ESM2 on seq_slice, return list of (did, h, attn_flat, true_ca, seq) dicts."""
    buf = []
    skipped = 0
    for did, seq in seq_slice:
        try:
            ids = tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)
            with torch.no_grad():
                out = esm(input_ids=ids, output_attentions=True)
            L        = len(seq)
            h        = out.last_hidden_state[0, 1:-1].float().unsqueeze(0)          # (1, L, 1280)
            attn     = torch.stack(out.attentions, dim=1)[0, :, :, 1:-1, 1:-1]     # (33, 20, L, L)
            attn_flat = attn.float().permute(2, 3, 0, 1).reshape(1, L, L, 660)     # (1, L, L, 660)
            true_ca  = ca_map[did].to(DEVICE)                                       # (L, 3)
            buf.append({"did": did, "seq": seq, "h": h, "attn_flat": attn_flat,
                        "true_ca": true_ca, "L": L})
        except Exception as e:
            log(f"  SKIP {did}: {e}"); skipped += 1
    return buf, skipped

# ── 11. Training loop ─────────────────────────────────────────────────────────
log(f"Training  N_STEPS={N_STEPS}  LR={LR}  buffer={BUFFER_SIZE}  steps_per_buf={STEPS_PER_BUFFER}")

pairs      = list(zip(did_list, seq_list))
random.shuffle(pairs)
pair_iter  = iter(pairs)
losses     = []
global_step = start_step
buf        = []
buf_fill_count = 0

while global_step < N_STEPS:
    # Refill buffer when empty
    if not buf:
        slice_pairs = []
        for _ in range(BUFFER_SIZE):
            try:
                slice_pairs.append(next(pair_iter))
            except StopIteration:
                random.shuffle(pairs)
                pair_iter = iter(pairs)
                slice_pairs.append(next(pair_iter))
        buf_fill_count += 1
        log(f"  Filling buffer {buf_fill_count} ({len(slice_pairs)} seqs) ...")
        buf, skipped = fill_buffer(slice_pairs)
        log(f"  Buffer ready: {len(buf)} sequences  (skipped {skipped})")
        if not buf:
            continue
        buf_step = 0

    # Sample from buffer
    item = buf[buf_step % len(buf)]
    buf_step += 1
    if buf_step >= STEPS_PER_BUFFER:
        buf = []  # trigger refill next iteration

    did      = item["did"]
    seq      = item["seq"]
    h        = item["h"]
    attn_flat = item["attn_flat"]
    true_ca  = item["true_ca"]
    L        = item["L"]

    try:
        aatype, seq_mask = get_aatype(seq)
    except Exception as e:
        log(f"  SKIP aatype {did}: {e}"); continue

    if attn_flat.shape[2] != L or true_ca.shape[0] != L:
        if global_step == 0:
            log(f"  Shape mismatch {did}: attn_flat L={attn_flat.shape[2]}, true_ca L={true_ca.shape[0]}, seq L={L}")
        continue

    mask   = seq_mask.unsqueeze(0)
    s_s    = fc_s(h)
    s_z    = fc_z(attn_flat)
    _, s_z_out = trunk(s_s, s_z, mask, num_recycling=RECYCLING)
    single     = sz_proj(s_z_out, s_s, mask[:, None, :] * mask[:, :, None])
    sm_out       = sm(s=single, z=s_z_out, aatype=aatype, mask=seq_mask)
    pred_ca      = sm_out["positions"][-1, 0, :, 1, :]
    plddt_logits = plddt_head(sm_out["single"])

    loss_struct = ca_dist_loss(pred_ca, true_ca)
    loss_plddt  = lddt_ce_loss(plddt_logits, pred_ca, true_ca)
    loss        = loss_struct + PLDDT_WEIGHT * loss_plddt
    if torch.isnan(loss) or torch.isinf(loss):
        if global_step == 0:
            log(f"  NaN/Inf loss on {did}: struct={loss_struct.item():.3f} plddt={loss_plddt.item():.3f}")
        continue

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())
    global_step += 1

    if global_step % 200 == 0:
        avg = sum(losses[-200:]) / len(losses[-200:])
        log(f"  step {global_step:5d}/{N_STEPS}  loss={avg:.3f}  "
            f"struct={loss_struct.item():.3f}  plddt={loss_plddt.item():.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}")

    if global_step % SAVE_EVERY == 0:
        ckpt = {"fc_s":       {k: v.cpu() for k,v in fc_s.state_dict().items()},
                "fc_z":       {k: v.cpu() for k,v in fc_z.state_dict().items()},
                "bb_update":  {k: v.cpu() for k,v in sm.bb_update.state_dict().items()},
                "plddt_head": {k: v.cpu() for k,v in plddt_head.state_dict().items()},
                "step": global_step, "backbone": "esm2_650m",
                "fc_s_in": 1280, "fc_z_in": 660, "fc_z_arch": "attention_660",
                "loss": sum(losses[-200:]) / max(1, len(losses[-200:]))}
        torch.save(ckpt, str(OUT_PATH.with_name(f"fcsz_fape_650m_step{global_step}.pt")))
        torch.save(ckpt, str(RESUME_PATH))
        log(f"  Checkpoint → fcsz_fape_650m_step{global_step}.pt")

avg = sum(losses) / max(1, len(losses))
log(f"Done  mean_loss={avg:.3f}")
torch.save({"fc_s":       {k: v.cpu() for k,v in fc_s.state_dict().items()},
            "fc_z":       {k: v.cpu() for k,v in fc_z.state_dict().items()},
            "bb_update":  {k: v.cpu() for k,v in sm.bb_update.state_dict().items()},
            "plddt_head": {k: v.cpu() for k,v in plddt_head.state_dict().items()},
            "step": global_step, "backbone": "esm2_650m",
            "fc_s_in": 1280, "fc_z_in": 660, "fc_z_arch": "attention_660", "loss": avg},
           str(OUT_PATH))
log(f"  Saved → {OUT_PATH}")
log(f"Total time: {(time.time()-T0)/60:.1f} min")
