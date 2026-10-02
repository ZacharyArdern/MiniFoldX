#!/usr/bin/env python3
"""
Colab: train fc_s + fc_z + FoldingTrunk + sz_proj with proper backbone FAPE loss.
Uses N/Ca/C coords from cath_s20_backbone.npz.
220 sequences held out (val_ids.txt); trains on train_ids.txt.

Two-phase LR: phase 1 (N steps at lr1), phase 2 (M steps at lr2).

Inputs:
    /content/weights/cath_s20_backbone.npz — CATH S20 N/Ca/C coords
    /content/weights/train_ids.txt         — 7584 training domain IDs
    /content/weights/val_ids.txt           — 220 held-out domain IDs
    /content/outputs/trunk_resume.pt       — optional resume checkpoint

Outputs:
    /content/outputs/trunk_best.pt         — best checkpoint
    /content/outputs/trunk_step<N>.pt      — intermediate checkpoints
    /content/outputs/trunk_final.pt        — final weights
"""

import os, random, time, sys, types, subprocess
from pathlib import Path

def run(*cmd):
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))

print("=== Installing dependencies ===", flush=True)
run(sys.executable, "-m", "pip", "install", "-q", "uv")
run("uv", "pip", "install", "--system", "-q",
    "git+https://github.com/ZacharyArdern/MiniFoldX.git#subdirectory=minifoldx/pytorch",
    "safetensors", "huggingface_hub[hf_xet]", "einops",
    "dm-tree", "ml-collections", "modelcif", "edit_distance", "fair-esm", "tmtools")

WEIGHTS_DIR  = Path("/content/weights");  WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR  = Path("/content/outputs");  OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RESUME_PATH  = OUTPUTS_DIR / "trunk_resume.pt"
FINAL_PATH   = OUTPUTS_DIR / "trunk_final.pt"
NPZ_PATH     = WEIGHTS_DIR / "cath_s20_backbone.npz"
TRAIN_IDS    = WEIGHTS_DIR / "train_ids.txt"
VAL_IDS      = WEIGHTS_DIR / "val_ids.txt"

# ── Config from env vars ──────────────────────────────────────────────────────
STEPS_PHASE1 = int(os.environ.get("STEPS_PHASE1", "95000"))
STEPS_PHASE2 = int(os.environ.get("STEPS_PHASE2", "5000"))
LR_PHASE1    = float(os.environ.get("LR_PHASE1", "1e-4"))
LR_PHASE2    = float(os.environ.get("LR_PHASE2", "1e-5"))
MAX_LEN      = int(os.environ.get("MAX_LEN", "180"))
RECYCLING    = int(os.environ.get("RECYCLING", "1"))
SAVE_EVERY   = int(os.environ.get("SAVE_EVERY", "5000"))
BOND_LAM     = float(os.environ.get("BOND_LAM", "0.1"))
SEED         = int(os.environ.get("SEED", "42"))

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {DEVICE}")

# ── ESM mock ──────────────────────────────────────────────────────────────────
import esm as _esm_mod
_dm = types.ModuleType("esm.data")
class _FA:
    @staticmethod
    def from_architecture(n): return _FA()
_dm.Alphabet = _FA
_esm_mod.data = _dm
sys.modules["esm.data"] = _dm

# ── Backbone / dims ───────────────────────────────────────────────────────────
BACKBONE   = os.environ.get("BACKBONE", "esmc_600m")
FC_S_IN    = {"esm2_650m": 1280, "esmc_600m": 1152}[BACKBONE]
FC_Z_IN    = {"esm2_650m": 660,  "esmc_600m": 648}[BACKBONE]
START_STEP = 0

ckpt_r = None
if RESUME_PATH.exists():
    log(f"Resuming from {RESUME_PATH}")
    ckpt_r     = torch.load(str(RESUME_PATH), map_location="cpu", weights_only=False)
    BACKBONE   = ckpt_r["backbone"]
    FC_S_IN    = ckpt_r["fc_s_in"]
    FC_Z_IN    = ckpt_r["fc_z_in"]
    START_STEP = ckpt_r.get("step", 0)
    log(f"  backbone={BACKBONE}  resumed_from_step={START_STEP}")
else:
    log(f"Starting fresh  backbone={BACKBONE}  fc_s_in={FC_S_IN}  fc_z_in={FC_Z_IN}")

# ── CATH S20 backbone coords ───────────────────────────────────────────────────
log(f"Loading {NPZ_PATH} ...")
npz      = np.load(str(NPZ_PATH), allow_pickle=True)
all_ids  = npz["domain_ids"].tolist()
all_seqs_map = dict(zip(all_ids, npz["sequences"].tolist()))
lengths  = npz["lengths"]
offsets  = np.concatenate([[0], np.cumsum(lengths)])
n_flat   = npz["n_coords_flat"]
ca_flat  = npz["ca_coords_flat"]
c_flat   = npz["c_coords_flat"]

backbone_map = {}
for i, did in enumerate(all_ids):
    s, e = offsets[i], offsets[i+1]
    backbone_map[did] = (
        torch.tensor(n_flat[s:e],  dtype=torch.float32),
        torch.tensor(ca_flat[s:e], dtype=torch.float32),
        torch.tensor(c_flat[s:e],  dtype=torch.float32),
    )

# ── Train / val split ─────────────────────────────────────────────────────────
if TRAIN_IDS.exists() and VAL_IDS.exists():
    with open(str(TRAIN_IDS)) as f:
        train_ids = [l.strip() for l in f if l.strip() and l.strip() in backbone_map]
    with open(str(VAL_IDS)) as f:
        val_ids = [l.strip() for l in f if l.strip() and l.strip() in backbone_map]
    log(f"  Train: {len(train_ids)}  Val: {len(val_ids)}")
else:
    log("WARNING: train_ids.txt/val_ids.txt not found — using all sequences for training")
    train_ids = list(backbone_map.keys())
    val_ids   = []

train_pairs = [(did, all_seqs_map[did]) for did in train_ids]

# ── MiniFoldX base weights ────────────────────────────────────────────────────
from safetensors.torch import load_file

CKPT_PATH = WEIGHTS_DIR / "minifold_12L.safetensors"
if not CKPT_PATH.exists():
    log("Downloading minifold_12L.safetensors ...")
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
    hf_hub_download(repo_id="z-ardern/MiniFoldX_weights",
                    filename="minifold_12L.safetensors",
                    local_dir=str(WEIGHTS_DIR), local_dir_use_symlinks=False)

sd = load_file(str(CKPT_PATH), device="cpu")
log("Base weights loaded")

from minifold.model.model import FoldingTrunk, PairToSequence
from minifold.model.structure import StructureModule
from minifold.model.heads import PerResidueLDDTCaPredictor
from minifold.data.config import model_config
from minifold.data.of_data import of_inference
from minifold.train.loss import backbone_loss, compute_plddt
from minifold.utils.rigid_utils import Rigid

# ── Build model ───────────────────────────────────────────────────────────────
log("Building model ...")
trunk = FoldingTrunk(c_s=1024, c_z=128, bins=32, disto_bins=64, num_layers=12, kernels=False)
trunk.load_state_dict(
    {k.replace("model.fold.", "").replace("miniformer._orig_mod.", "miniformer."): v
     for k, v in sd.items() if k.startswith("model.fold.")}, strict=False)
if ckpt_r: trunk.load_state_dict(ckpt_r["trunk"])
trunk = trunk.to(DEVICE)

sz_proj = PairToSequence(c_z=128, c_s=1024, c_s_out=1024)
sz_proj.load_state_dict({k.replace("model.sz_project.", ""): v
                          for k, v in sd.items() if "sz_project" in k})
if ckpt_r: sz_proj.load_state_dict(ckpt_r["sz_proj"])
sz_proj = sz_proj.to(DEVICE)

sm = StructureModule(c_s=1024, c_z=128, c_resnet=128, head_dim=64, no_heads=16,
                     no_blocks=8, no_resnet_blocks=2, no_angles=7,
                     trans_scale_factor=10, epsilon=1e-5, inf=1e5)
sm.load_state_dict({k.replace("model.structure_module.", ""): v
                    for k, v in sd.items() if "structure_module" in k})
sm = sm.to(DEVICE).eval()
for p in sm.parameters(): p.requires_grad_(False)

plddt_head = PerResidueLDDTCaPredictor(no_bins=50, c_in=1024, c_hidden=128)
plddt_head.load_state_dict({k.replace("model.aux_heads.plddt.", ""): v
                             for k, v in sd.items() if "aux_heads.plddt" in k})
plddt_head = plddt_head.to(DEVICE).eval()
for p in plddt_head.parameters(): p.requires_grad_(False)

fc_s = nn.Sequential(nn.Linear(FC_S_IN, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(DEVICE)
fc_z = nn.Sequential(nn.Linear(FC_Z_IN, 128),  nn.ReLU(), nn.Linear(128,  128)).to(DEVICE)
if ckpt_r:
    fc_s.load_state_dict(ckpt_r["fc_s"])
    fc_z.load_state_dict(ckpt_r["fc_z"])

# ── Load backbone ─────────────────────────────────────────────────────────────
if BACKBONE == "esm2_650m":
    from transformers import EsmModel, EsmTokenizer
    log("Loading ESM2-650M ...")
    _tok = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    _esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                     attn_implementation="eager").half().to(DEVICE).eval()
    for p in _esm.parameters(): p.requires_grad_(False)

    def encode(seq):
        ids = _tok(seq, return_tensors="pt")["input_ids"].to(DEVICE)
        with torch.no_grad():
            out = _esm(input_ids=ids, output_attentions=True)
        L = len(seq)
        h         = out.last_hidden_state[0, 1:-1].float().unsqueeze(0)
        attn      = torch.stack(out.attentions, dim=1)[0, :, :, 1:-1, 1:-1]
        attn_flat = attn.float().permute(2, 3, 0, 1).reshape(1, L, L, FC_Z_IN)
        return h, attn_flat

elif BACKBONE == "esmc_600m":
    from esm.models.esmc import ESMC, EsmSequenceTokenizer
    log("Loading ESMC-600M ...")
    _esmc_tok = EsmSequenceTokenizer()
    _esmc = ESMC.from_pretrained("esmc_600m").to(DEVICE).eval()
    for p in _esmc.parameters(): p.requires_grad_(False)

    def encode(seq):
        tok_out   = _esmc_tok([seq])
        ids       = torch.tensor(tok_out["input_ids"], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out   = _esmc(sequence_tokens=ids, output_attentions=True)
        L         = len(seq)
        h         = out.embeddings[:, 1:-1].float()
        attns     = torch.stack(out.attentions, dim=1)[:, :, :, 1:-1, 1:-1].float()
        attn_flat = attns.permute(0, 3, 4, 1, 2).reshape(1, L, L, FC_Z_IN)
        return h, attn_flat

else:
    raise ValueError(f"Unknown backbone: {BACKBONE}")

log(f"  {BACKBONE} ready")

cfg = model_config("initial_training", train=False, low_prec=False,
                   long_sequence_inference=False).data

def get_aatype(seq):
    f = of_inference(seq, "predict", cfg)
    return f["aatype"][:, 0].to(DEVICE), f["seq_mask"][:, 0].float().to(DEVICE)

# ── Optimiser ─────────────────────────────────────────────────────────────────
trainable = (list(fc_s.parameters()) + list(fc_z.parameters()) +
             list(trunk.parameters()) + list(sz_proj.parameters()))
n_p = sum(p.numel() for p in trainable)
log(f"Trainable: {n_p/1e6:.2f}M params  bond_lam={BOND_LAM}")

def train_phase(phase, n_steps, lr):
    log(f"=== Phase {phase}: {n_steps} steps at lr={lr} ===")
    opt = torch.optim.Adam(trainable, lr=lr)
    trunk.train(); sz_proj.train(); fc_s.train(); fc_z.train()

    random.seed(SEED + phase)
    torch.manual_seed(SEED + phase)

    step = 0
    loss_buf, fape_buf, bond_buf = [], [], []
    best_loss = float("inf")
    t0 = time.time()

    while step < n_steps:
        pairs = list(train_pairs)
        random.shuffle(pairs)
        for did, seq in pairs:
            if step >= n_steps:
                break
            try:
                L = len(seq)
                h, attn_flat = encode(seq)
                n_c, ca_c, c_c = backbone_map[did]
                if ca_c.shape[0] != L: continue

                # Ground-truth backbone frames
                n_t  = n_c.unsqueeze(0).to(DEVICE)
                ca_t = ca_c.unsqueeze(0).to(DEVICE)
                c_t  = c_c.unsqueeze(0).to(DEVICE)
                gt_rigid = Rigid.from_3_points(
                    p_neg_x_axis=n_t, origin=ca_t, p_xy_plane=c_t)
                bb_rigid_tensor = gt_rigid.to_tensor_4x4()

                aatype, seq_mask = get_aatype(seq)
                mask = seq_mask.unsqueeze(0)

                s_s = fc_s(h)
                s_z = fc_z(attn_flat)
                _, s_z_out = trunk(s_s, s_z, mask, num_recycling=RECYCLING)
                single     = sz_proj(s_z_out, s_s, mask[:, None, :] * mask[:, :, None])
                sm_out     = sm(s=single, z=s_z_out, aatype=aatype, mask=seq_mask)

                fape = backbone_loss(
                    backbone_rigid_tensor=bb_rigid_tensor,
                    backbone_rigid_mask=mask,
                    traj=sm_out["frames"],
                    clamp_distance=10.0,
                    loss_unit_distance=10.0,
                )
                pred_ca   = sm_out["positions"][-1, 0, :, 1, :]
                bond_loss = (((pred_ca[1:] - pred_ca[:-1]).norm(dim=-1) - 3.8) ** 2).mean()
                loss      = fape + BOND_LAM * bond_loss

                if torch.isnan(loss) or torch.isinf(loss): continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

                step += 1
                loss_buf.append(loss.item())
                fape_buf.append(fape.item())
                bond_buf.append(bond_loss.item())

            except Exception as e:
                import traceback; traceback.print_exc()
                continue

            if step % 200 == 0:
                n = 200
                sps = n / (time.time() - t0)
                eta = (n_steps - step) / sps / 3600
                log(f"  p{phase} step {step:6d}/{n_steps}  "
                    f"loss={np.mean(loss_buf[-n:]):.4f}  "
                    f"fape={np.mean(fape_buf[-n:]):.4f}  "
                    f"bond={np.mean(bond_buf[-n:]):.4f}  "
                    f"sps={sps:.1f}  eta={eta:.1f}h")
                t0 = time.time()

            if step % SAVE_EVERY == 0 or step == n_steps:
                global_step = START_STEP + (STEPS_PHASE1 if phase == 2 else 0) + step
                cur_loss = float(np.mean([x for x in loss_buf[-SAVE_EVERY:] if not np.isnan(x)] or [float("nan")]))
                ckpt = dict(step=global_step, loss=cur_loss, backbone=BACKBONE,
                            fc_s_in=FC_S_IN, fc_z_in=FC_Z_IN,
                            fc_s=fc_s.state_dict(), fc_z=fc_z.state_dict(),
                            trunk=trunk.state_dict(), sz_proj=sz_proj.state_dict())
                step_path = OUTPUTS_DIR / f"trunk_step{global_step}.pt"
                torch.save(ckpt, str(step_path))
                log(f"  Checkpoint → {step_path.name}")
                if not np.isnan(cur_loss) and cur_loss < best_loss:
                    best_loss = cur_loss
                    tmp = OUTPUTS_DIR / "trunk_best.tmp"
                    torch.save(ckpt, str(tmp))
                    tmp.replace(OUTPUTS_DIR / "trunk_best.pt")
                    log(f"  → new best loss={best_loss:.4f}")

    log(f"Phase {phase} done. Mean loss: {np.mean(loss_buf):.4f}")
    return loss_buf

# ── Run phases ────────────────────────────────────────────────────────────────
losses1 = train_phase(1, STEPS_PHASE1, LR_PHASE1)
losses2 = train_phase(2, STEPS_PHASE2, LR_PHASE2)

# ── Save final ────────────────────────────────────────────────────────────────
global_step = START_STEP + STEPS_PHASE1 + STEPS_PHASE2
all_losses  = losses1 + losses2
final_ckpt  = dict(
    step=global_step, loss=float(np.mean(all_losses)),
    backbone=BACKBONE, fc_s_in=FC_S_IN, fc_z_in=FC_Z_IN,
    fc_s=fc_s.state_dict(), fc_z=fc_z.state_dict(),
    trunk=trunk.state_dict(), sz_proj=sz_proj.state_dict(),
)
torch.save(final_ckpt, str(FINAL_PATH))
log(f"Final checkpoint → {FINAL_PATH}  (global_step={global_step})")
log(f"Total time: {(time.time()-T0)/60:.1f} min")
