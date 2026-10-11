import os
import sys
import types
import urllib.request
from pathlib import Path
from typing import Optional

# The new ESM package (v3+) dropped esm.data. Mock it before any minifold imports
# so that type annotations in minifold/utils/esm.py resolve correctly.
def _mock_esm_data():
    import esm as _esm_mod
    if hasattr(_esm_mod, "data"):
        return
    _dm = types.ModuleType("esm.data")
    class _Alphabet:
        padding_idx = 1
        mask_idx = 32
        cls_idx = 0
        eos_idx = 2
        prepend_bos = True
        append_eos = True
        @staticmethod
        def from_architecture(n): return _Alphabet()
        def get_batch_converter(self): return lambda x: (None, None, None)
        def encode(self, seq): return list(range(len(seq)))
        def __len__(self): return 33
    _dm.Alphabet = _Alphabet
    _esm_mod.data = _dm
    sys.modules["esm.data"] = _dm
_mock_esm_data()

import click
import torch
import torch.nn.functional as F
import numpy as np
from Bio import SeqIO

from minifold.model.model import MiniFoldModel
from minifold.utils.residue_constants import restype_order_with_x
from minifold.utils.protein import Protein, to_pdb
from minifold.data.config import model_config
from minifold.data.of_data import of_inference
from minifold.utils.residue_constants import restype_order_with_x_inverse


MODEL_URL_48L = (
    "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt"
)

MODEL_URL_12L = (
    "https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_12L_final.ckpt"
)


def download(cache: Path, model_size: str) -> None:
    model = cache / f"minifold_{model_size}.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        url = MODEL_URL_48L if model_size == "48L" else MODEL_URL_12L
        urllib.request.urlretrieve(url, str(model))  # noqa: S310


def output_to_pdb(seq, coords, mask, plddt):
    mapping = restype_order_with_x
    af_seq = np.array([mapping[res] for res in seq])
    plddt = plddt[:, None].repeat(coords.shape[1], axis=1)
    pred = Protein(
        aatype=af_seq,
        atom_positions=coords,
        atom_mask=mask,
        residue_index=1 + np.arange(len(seq)),
        b_factors=plddt,
    )
    return to_pdb(pred)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


_HPARAMS = {
    "48L": {"esm_model_name": "esm2_t36_3B_UR50D", "num_blocks": 48, "no_bins": 64},
    "12L": {"esm_model_name": "esm2_t36_3B_UR50D", "num_blocks": 12, "no_bins": 64},
}


def _quantize_esm2_int8(module) -> None:
    import bitsandbytes as bnb
    import torch.nn as nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            q = bnb.nn.Linear8bitLt(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            q.weight = bnb.nn.Int8Params(
                child.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            )
            if child.bias is not None:
                import torch.nn as nn2
                q.bias = nn2.Parameter(child.bias.data.clone())
            setattr(module, name, q)
        else:
            _quantize_esm2_int8(child)


def create_model(checkpoint, device, compile=False, kernels=False, int8_esm2=False):
    checkpoint = str(checkpoint)
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint, device="cpu")
        size = "48L" if "48L" in checkpoint else "12L"
        hparams = _HPARAMS[size]
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]

    if kernels:
        torch._dynamo.config.cache_size_limit = 64

    config_of = model_config(
        "initial_training",
        train=False,
        low_prec=False,
        long_sequence_inference=False,
    )
    model = MiniFoldModel(
        esm_model_name=hparams["esm_model_name"],
        num_blocks=hparams["num_blocks"],
        no_bins=hparams["no_bins"],
        config_of=config_of,
        use_structure_module=True,
        kernels=kernels,
    )
    if device.type != "cpu":
        model.lm = model.lm.half()

    from esm.data import Alphabet
    alphabet = Alphabet.from_architecture("ESM-1b")

    state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
    state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    if int8_esm2:
        print("Quantizing ESM2 to int8 (bitsandbytes) ...", flush=True)
        _quantize_esm2_int8(model.lm)

    if compile:
        model.fold.miniformer = torch.compile(
            model.fold.miniformer,
            dynamic=True,
            fullgraph=True,
        )

    model = model.to(device)
    model.eval()

    return alphabet, model


def create_esmc_model(checkpoint, device):
    """Load ESMC-600M backbone + trained trunk from a fine-tuned .pt checkpoint."""
    import glob
    import torch.nn as nn
    from safetensors.torch import load_file
    from minifold.model.model import FoldingTrunk, PairToSequence
    from minifold.model.structure import StructureModule
    from minifold.model.heads import PerResidueLDDTCaPredictor
    from minifold.train.loss import compute_plddt

    # Base MiniFoldX weights (structure module, pLDDT head)
    cache_dir = Path.home() / ".cache" / "minifoldx"
    sd = load_file(str(cache_dir / "minifold_12L.safetensors"), device="cpu")

    # Load trained trunk checkpoint
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    fc_s_in = ckpt["fc_s_in"]
    fc_z_in = ckpt["fc_z_in"]
    print(f"  Backbone: {ckpt.get('backbone')}  fc_s_in={fc_s_in}  fc_z_in={fc_z_in}", flush=True)

    trunk = FoldingTrunk(c_s=1024, c_z=128, bins=32, disto_bins=64, num_layers=12, kernels=False)
    trunk.load_state_dict(
        {k.replace("model.fold.", "").replace("miniformer._orig_mod.", "miniformer."): v
         for k, v in sd.items() if k.startswith("model.fold.")}, strict=False)
    trunk.load_state_dict(ckpt["trunk"])
    trunk = trunk.to(device).eval()
    for p in trunk.parameters(): p.requires_grad_(False)

    sz_proj = PairToSequence(c_z=128, c_s=1024, c_s_out=1024)
    sz_proj.load_state_dict(
        {k.replace("model.sz_project.", ""): v for k, v in sd.items() if "sz_project" in k})
    sz_proj.load_state_dict(ckpt["sz_proj"])
    sz_proj = sz_proj.to(device).eval()
    for p in sz_proj.parameters(): p.requires_grad_(False)

    sm = StructureModule(c_s=1024, c_z=128, c_resnet=128, head_dim=64, no_heads=16,
                         no_blocks=8, no_resnet_blocks=2, no_angles=7,
                         trans_scale_factor=10, epsilon=1e-5, inf=1e5)
    sm.load_state_dict(
        {k.replace("model.structure_module.", ""): v for k, v in sd.items() if "structure_module" in k})
    sm = sm.to(device).eval()
    for p in sm.parameters(): p.requires_grad_(False)

    plddt_head = PerResidueLDDTCaPredictor(no_bins=50, c_in=1024, c_hidden=128)
    plddt_head.load_state_dict(
        {k.replace("model.aux_heads.plddt.", ""): v for k, v in sd.items() if "aux_heads.plddt" in k})
    plddt_head = plddt_head.to(device).eval()
    for p in plddt_head.parameters(): p.requires_grad_(False)

    fc_s = nn.Sequential(nn.Linear(fc_s_in, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(device).eval()
    fc_z = nn.Sequential(nn.Linear(fc_z_in, 128),  nn.ReLU(), nn.Linear(128, 128)).to(device).eval()
    fc_s.load_state_dict(ckpt["fc_s"])
    fc_z.load_state_dict(ckpt["fc_z"])
    for p in list(fc_s.parameters()) + list(fc_z.parameters()): p.requires_grad_(False)

    # Load ESMC-600M
    from esm.models.esmc import ESMC, EsmSequenceTokenizer
    tok = EsmSequenceTokenizer()
    pth_files = glob.glob(str(
        Path.home() / ".cache/huggingface/hub/models--biohub--esmc-600m-2024-12/snapshots/*/data/weights/*.pth"))
    if not pth_files:
        raise FileNotFoundError("ESMC-600M weights not found in HF cache.")
    esmc = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=tok, use_flash_attn=False)
    esmc.load_state_dict(torch.load(pth_files[0], map_location="cpu"), strict=False)
    esmc = esmc.to(device).eval()
    for p in esmc.parameters(): p.requires_grad_(False)

    return dict(tok=tok, esmc=esmc, fc_s=fc_s, fc_z=fc_z, fc_z_in=fc_z_in,
                trunk=trunk, sz_proj=sz_proj, sm=sm, plddt_head=plddt_head,
                compute_plddt=compute_plddt)


def encode_batch_esmc(seqs, components, device):
    """Encode a batch of same-length sequences with ESMC, return list of (h, attn_flat)."""
    tok = components["tok"]
    esmc = components["esmc"]
    fc_z_in = components["fc_z_in"]

    tok_out = tok(seqs)
    ids = torch.tensor(tok_out["input_ids"], dtype=torch.long, device=device)
    with torch.no_grad():
        out = esmc(sequence_tokens=ids, output_attentions=True)

    results = []
    for i, seq in enumerate(seqs):
        L = len(seq)
        h = out.embeddings[i:i+1, 1:L+1].float()
        attns = torch.stack(out.attentions, dim=1)[i:i+1, :, :, 1:L+1, 1:L+1].float()
        attn_flat = attns.permute(0, 3, 4, 1, 2).reshape(1, L, L, fc_z_in)
        results.append((h, attn_flat))
    return results


def create_batches(fasta, token_per_batch):
    import gzip
    opener = gzip.open if str(fasta).endswith(".gz") else open
    with opener(fasta, "rt") as fh:
        seqs = list(SeqIO.parse(fh, "fasta"))
    seqs = sorted(seqs, key=lambda x: len(str(x.seq)))

    batches = []
    while len(seqs) > 0:
        seq = seqs.pop(0)
        batch = [seq]

        while True:
            if len(seqs) == 0:
                break
            next_seq = seqs[0]
            next_batch_size = (len(batch) + 1) * len(str(next_seq.seq))
            if next_batch_size > token_per_batch:
                break
            seq = seqs.pop(0)
            batch.append(seq)

        batches.append(batch)
    return batches


def create_esmc_encode_batches(seqs, max_encode_tokens=600):
    """Group sequences by length for batched ESMC encoding.

    Sequences are sorted by length and grouped so that
    batch_size * max_len_in_batch <= max_encode_tokens.
    Each returned batch is a list of (name, seq) pairs.
    """
    sorted_seqs = sorted(seqs, key=lambda x: len(x[1]))
    batches = []
    i = 0
    while i < len(sorted_seqs):
        batch = [sorted_seqs[i]]
        max_l = len(sorted_seqs[i][1])
        i += 1
        while i < len(sorted_seqs):
            candidate_l = len(sorted_seqs[i][1])
            if (len(batch) + 1) * candidate_l > max_encode_tokens:
                break
            batch.append(sorted_seqs[i])
            max_l = candidate_l
            i += 1
        batches.append(batch)
    return batches


def prepare_input(seq, config, alphabet):
    open_fold_batch = of_inference(seq, "predict", config)
    of_seq = "".join(
        [restype_order_with_x_inverse[x.item()] for x in open_fold_batch["aatype"]]
    )[: open_fold_batch["seq_length"]]
    encoded_seq = alphabet.encode(of_seq)
    encoded_seq = torch.tensor(encoded_seq, dtype=torch.long)
    mask = open_fold_batch["seq_mask"][:, 0].bool()
    relevant = {"aatype", "seq_mask", "residx_atom37_to_atom14", "atom37_atom_exists"}
    open_fold_batch = {k: v for k, v in open_fold_batch.items() if k in relevant}
    return encoded_seq, mask, open_fold_batch


@click.command()
@click.argument("fasta", type=click.Path(exists=True))
@click.option("--out_dir", type=click.Path(exists=False),
              help="Path to save predictions.", default="./minifold_predictions")
@click.option("--cache", type=click.Path(exists=False),
              help="Directory for downloaded data and model.", default="./minifold_cache")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint path. Required for esmc_600m; optional for esm2_3b.",
              default=None)
@click.option("--token_per_batch", type=int, default=2048,
              help="Tokens per batch for esm2_3b. Default 2048.")
@click.option("--compile", is_flag=True, help="Compile the model (esm2_3b only).")
@click.option("--model_size", type=click.Choice(["48L", "12L"]), default="48L",
              help="Model size for esm2_3b. Default 48L.")
@click.option("--kernels", is_flag=True, help="Use custom kernels (esm2_3b only).")
@click.option("--num_recycling", type=int, default=0,
              help="Number of recycling iterations. Default 0.")
@click.option("--int8-esm2", "int8_esm2", is_flag=True,
              help="Quantize ESM2 to int8 (esm2_3b only).")
@click.option("--cpu", "force_cpu", is_flag=True,
              help="Force CPU inference.")
@click.option("--esm_model", type=click.Choice(["esmc_600m", "esm2_3b"]),
              default="esm2_3b", show_default=True,
              help="Backbone model. esmc_600m requires --checkpoint.")
@click.option("--max_encode_tokens", type=int, default=600,
              help="Token budget for batched ESMC encoding (esmc_600m only). Default 600.")
@click.option("--early_stop", type=float, default=None,
              help="Stop recycling when mean pLDDT >= this value (e.g. 85). Default: disabled.")
def predict(
    fasta: str,
    out_dir: str = "./minifold_predictions",
    cache: str = "./minifold_cache",
    checkpoint: Optional[str] = None,
    token_per_batch: int = 2048,
    compile: bool = False,
    model_size: str = "48L",
    kernels: bool = False,
    num_recycling: int = 3,
    int8_esm2: bool = False,
    force_cpu: bool = False,
    esm_model: str = "esmc_600m",
    max_encode_tokens: int = 600,
    early_stop: Optional[float] = None,
) -> None:
    """Run predictions with Minifold."""
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    fasta = Path(fasta).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"minifold_results_{fasta.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if force_cpu:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count() or 1)
    else:
        device = get_device()
    print(f"Using device: {device}  backbone: {esm_model}", flush=True)

    torch.hub.set_dir(cache)
    torch.set_grad_enabled(False)

    config = model_config("initial_training", train=False, low_prec=False,
                          long_sequence_inference=False).data

    if esm_model == "esmc_600m":
        _predict_esmc(fasta, out_dir, checkpoint, device, config,
                      num_recycling, max_encode_tokens, early_stop)
    else:
        _predict_esm2_3b(fasta, out_dir, cache, checkpoint, device, config,
                         token_per_batch, compile, model_size, kernels,
                         num_recycling, int8_esm2, early_stop)


def _predict_esmc(fasta, out_dir, checkpoint, device, config, num_recycling, max_encode_tokens, early_stop=None):
    if checkpoint is None:
        raise click.UsageError("--checkpoint is required for esmc_600m.")

    print("Load model...", flush=True)
    components = create_esmc_model(checkpoint, device)

    import gzip
    from minifold.utils.feats import atom14_to_atom37
    opener = gzip.open if str(fasta).endswith(".gz") else open
    with opener(str(fasta), "rt") as fh:
        all_seqs = [(str(r.id), str(r.seq)) for r in SeqIO.parse(fh, "fasta")]

    print(f"Load sequences... {len(all_seqs)} sequence(s)", flush=True)

    # Group into encode batches by length
    encode_batches = create_esmc_encode_batches(all_seqs, max_encode_tokens)
    n_seqs = len(all_seqs)
    n_batches = len(encode_batches)
    print(f"Launching predictions: {n_seqs} sequence(s) in {n_batches} encode batch(es)...",
          flush=True)

    completed = 0
    for batch_idx, batch in enumerate(encode_batches, 1):
        lengths = [len(s) for _, s in batch]
        print(f"\n[{batch_idx}/{n_batches}] {len(batch)} seq(s), "
              f"L={min(lengths)}-{max(lengths)} ({completed}/{n_seqs} done)", flush=True)

        seqs_only = [s for _, s in batch]
        try:
            encoded = encode_batch_esmc(seqs_only, components, device)
        except Exception as e:
            print(f"  Encode failed: {e}")
            import traceback; traceback.print_exc()
            completed += len(batch)
            continue

        for (name, seq), (h, attn_flat) in zip(batch, encoded):
            try:
                L = len(seq)
                f = of_inference(seq, "predict", config)
                aatype   = f["aatype"][:, 0].to(device)
                seq_mask = f["seq_mask"][:, 0].float().to(device)
                mask = seq_mask.unsqueeze(0)

                s_s = components["fc_s"](h)
                s_z = components["fc_z"](attn_flat)
                pair_mask = mask[:, None, :] * mask[:, :, None]
                if early_stop is not None:
                    from minifold.train.loss import compute_plddt as _cplddt
                    s_z_proj, pm_f, _ = components["trunk"]._project(s_s, s_z, mask[:, 0, :] if mask.dim() == 3 else mask)
                    shape = tuple(s_z_proj.shape[:3]) + (components["trunk"].disto_bins,)
                    dists = torch.zeros(shape, device=s_z_proj.device, dtype=s_z_proj.dtype)
                    s_z_out = None
                    for _i in range(num_recycling + 1):
                        _, s_z_out, dists = components["trunk"]._single_pass(s_z_proj, pm_f, dists)
                        _single = components["sz_proj"](s_z_out, s_s, pm_f)
                        _sm = components["sm"](s=_single, z=s_z_out, aatype=aatype, mask=seq_mask)
                        _mplddt = float(_cplddt(components["plddt_head"](_sm["single"])).mean())
                        print(f"  pass {_i+1}/{num_recycling+1}: pLDDT={_mplddt:.1f}", flush=True)
                        single, sm_out = _single, _sm
                        if _mplddt >= early_stop and _i < num_recycling:
                            print(f"  → early stop (>= {early_stop})", flush=True)
                            break
                else:
                    _, s_z_out = components["trunk"](s_s, s_z, mask, num_recycling=num_recycling)
                    single = components["sz_proj"](s_z_out, s_s, pair_mask)
                    sm_out = components["sm"](s=single, z=s_z_out, aatype=aatype, mask=seq_mask)

                of_feats = {
                    "aatype":                    aatype.unsqueeze(0),
                    "seq_mask":                  seq_mask.unsqueeze(0),
                    "residx_atom37_to_atom14":   f["residx_atom37_to_atom14"][:, :, 0].unsqueeze(0).to(device),
                    "atom37_atom_exists":        f["atom37_atom_exists"][:, :, 0].unsqueeze(0).to(device),
                }
                final_pos  = atom14_to_atom37(sm_out["positions"][-1], of_feats)
                final_mask = of_feats["atom37_atom_exists"]

                plddt_logits = components["plddt_head"](sm_out["single"])
                plddt_per_res = components["compute_plddt"](plddt_logits).float().cpu().numpy()[0]

                pos_np  = final_pos.float().cpu().numpy()[0]
                mask_np = final_mask.float().cpu().numpy()[0]
                prot = Protein(
                    atom_positions=pos_np,
                    aatype=aatype.cpu().numpy(),
                    atom_mask=mask_np,
                    residue_index=np.arange(1, L + 1),
                    b_factors=np.broadcast_to(plddt_per_res[:, None], (L, 37)),
                )
                pdb_path = out_dir / f"{name}.pdb"
                pdb_path.write_text(to_pdb(prot))
                print(f"  {name}: L={L}  pLDDT={plddt_per_res.mean():.1f}  → {pdb_path.name}",
                      flush=True)
            except Exception as e:
                print(f"  {name}: FAILED — {e}")
                import traceback; traceback.print_exc()

            completed += 1

    print(f"\nDone. {completed}/{n_seqs} structure(s) saved to {out_dir}", flush=True)


def _predict_esm2_3b(fasta, out_dir, cache, checkpoint, device, config,
                     token_per_batch, compile, model_size, kernels,
                     num_recycling, int8_esm2, early_stop=None):
    if checkpoint is None:
        download(cache, model_size)
        checkpoint = cache / f"minifold_{model_size}.ckpt"

    print("Load model...", flush=True)
    alphabet, model = create_model(checkpoint, device, compile, kernels=kernels,
                                   int8_esm2=int8_esm2)
    if device.type == "cpu":
        model = model.float()

    print("Load sequences...", flush=True)
    batches = create_batches(fasta, token_per_batch)
    n_batches = len(batches)
    n_seqs = sum(len(b) for b in batches)
    print(f"Launching predictions: {n_seqs} sequence(s) in {n_batches} batch(es)...",
          flush=True)

    from minifold.utils.feats import atom14_to_atom37
    completed = 0
    for batch_idx, batch in enumerate(batches, 1):
        max_seq_len = max(len(str(t.seq)) for t in batch)
        print(f"\n[{batch_idx}/{n_batches}] {len(batch)} seq(s), max len {max_seq_len} aa "
              f"({completed}/{n_seqs} done)", flush=True)

        feats = [prepare_input(str(t.seq), config, alphabet) for t in batch]
        max_len = max(len(seq) for seq, _, _ in feats)
        if kernels:
            max_len = (max_len + 127) // 128 * 128

        seq = torch.stack(
            [F.pad(s, (0, max_len - len(s)), value=20) for s, _, _ in feats])
        mask = torch.stack(
            [F.pad(m, (0, max_len - len(m)), value=0) for _, m, _ in feats])
        batch_of = {}
        for _, _, feats_of in feats:
            for k, v in feats_of.items():
                batch_of.setdefault(k, []).append(v)
        for k, v in batch_of.items():
            batch_of[k] = torch.stack([
                F.pad(item, [0] * 2 * (len(item.shape) - 1) + [0, max_len - item.shape[0]], value=0)
                for item in v])

        model_batch = {
            "seq": seq.to(device),
            "mask": mask.to(device),
            "batch_of": {k: v.to(device) for k, v in batch_of.items()},
        }

        try:
            import contextlib
            if device.type == "cpu":
                autocast_ctx = contextlib.nullcontext()
            else:
                autocast_device = "cuda" if device.type == "cuda" else "mps"
                autocast_ctx = torch.autocast(autocast_device, dtype=torch.bfloat16)
            with autocast_ctx:
                out = model(model_batch, num_recycling=num_recycling,
                            early_stop=early_stop)
                out_pos  = out.pop("final_atom_positions")
                out_mask = out.pop("final_atom_mask")
                plddt    = out.pop("plddt")
                del out

            out_pos  = out_pos.float().cpu().numpy()
            out_mask = out_mask.float().cpu().numpy()
            plddt    = plddt.float().cpu().numpy()

            for idx, t in enumerate(batch):
                path_out = out_dir / f"{str(t.id)}.pdb"
                pdb_string = output_to_pdb(
                    str(t.seq),
                    out_pos[idx,  : len(t.seq)],
                    out_mask[idx, : len(t.seq)],
                    plddt[idx,    : len(t.seq)],
                )
                with open(path_out, "w") as f:
                    f.write(pdb_string)

            completed += len(batch)
            print(f"  → saved {len(batch)} structure(s) ({completed}/{n_seqs} total)",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to predict batch: {e}")


if __name__ == "__main__":
    predict()
