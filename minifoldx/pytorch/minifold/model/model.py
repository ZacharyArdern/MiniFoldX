import torch
import torch.nn as nn

from minifold.model.miniformer import MiniFormer
from minifold.utils.tensor_utils import tensor_tree_map
from minifold.utils.feats import atom14_to_atom37
from minifold.model.heads import AuxiliaryHeads
from minifold.model.structure import StructureModule
from minifold.utils.esm import load_model_and_alphabet


class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2 * inner_dim
        """
        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.
        diff[mask == 0] = 0
        output = self.embedding(diff)
        return output


class PairToSequence(nn.Module):
    def __init__(self, c_z=128, c_s=1024, c_s_out=1024):
        super().__init__()

        self.s_z_mlp = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
        )
        self.combiner = nn.Sequential(
            nn.Linear(2 * c_z + c_s, c_s_out),
        )

    def forward(self, s_z, s_s_in, pair_mask):
        # MLP
        s_z = self.s_z_mlp(s_z)

        # s_z -> s_s
        s_z = s_z * pair_mask[..., None]

        # Column average
        norm = pair_mask.sum(dim=2).clamp(min=1)
        s_s_c = s_z.sum(dim=2) / norm[..., None]

        # Row average
        norm = pair_mask.sum(dim=1).clamp(min=1)
        s_s_r = s_z.sum(dim=1) / norm[..., None]

        # Combine with initial s_s
        s_s = self.combiner(torch.cat([s_s_c, s_s_r, s_s_in], dim=-1))
        return s_s


class FoldingTrunk(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        bins,
        disto_bins=64,
        num_layers=1,
        kernels=False,
    ):
        super().__init__()
        self.disto_bins = disto_bins
        self.positional_embedding = RelativePosition(bins, c_z)
        self.seq_to_pair = SequenceToPair(c_s, c_z // 2, c_z)
        self.projection = nn.Linear(c_z * 3, c_z)  # here out c_z * 2
        self.recycle = nn.Linear(disto_bins, c_z)  # here
        self.miniformer = MiniFormer(c_z, blocks=num_layers, kernels=kernels)
        self.fc_out = nn.Sequential(
            # nn.LayerNorm(c_z * 2),  # here
            nn.Linear(c_z, c_z),  # here
            nn.ReLU(),
            nn.Linear(c_z, disto_bins),
        )
        torch.nn.init.zeros_(self.seq_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.seq_to_pair.o_proj.bias)

    def forward(self, s_s, s_z, mask, num_recycling=0):
        """
        Inputs:
          s_s_0:     B x L x C            tensor of sequence features
          s_z_0:    B x L x L x C        tensor of pair features
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """
        # Make pairwise mask
        pair_mask = mask[:, None, :] * mask[:, :, None]

        # Add positional embeddings
        residx = torch.arange(s_s.shape[1], device=s_s.device)
        residx = residx.unsqueeze(0).expand(s_s.shape[0], -1)

        # Concatenate and project
        s_z = torch.cat(
            [
                s_z,
                self.seq_to_pair(s_s),
                self.positional_embedding(residx, mask=pair_mask),
            ],
            dim=-1,
        )
        s_z = self.projection(s_z)

        # Set masks to floats
        mask = mask.to(s_z)
        pair_mask = pair_mask.to(s_z)

        # Initialize binned distance
        shape = tuple(s_z.shape[:3]) + (self.disto_bins,)
        dists = torch.zeros(shape, device=s_z.device, dtype=s_z.dtype)

        # Perform folding rounds
        for i in range(num_recycling + 1):
            with torch.set_grad_enabled(self.training and (i == num_recycling)):
                # Issue with unused parameters in autocast
                if (
                    self.training
                    and (i == num_recycling)
                    and torch.is_autocast_enabled()
                ):
                    torch.clear_autocast_cache()

                # Compute blocks
                s_z_c = s_z + self.recycle(dists)
                s_z_c = self.miniformer(s_z_c, pair_mask)

                # Output MLP
                preds = self.fc_out(s_z_c + s_z_c.transpose(1, 2))

                # Compute binned distance for recycling
                dists = preds.detach().argmax(dim=-1)
                dists = nn.functional.one_hot(dists, self.disto_bins).to(s_z)

        return preds, s_z_c

    def _project(self, s_s, s_z, mask):
        """One-time projection before the recycling loop (for early-stop path)."""
        pair_mask = mask[:, None, :] * mask[:, :, None]
        residx = torch.arange(s_s.shape[1], device=s_s.device)
        residx = residx.unsqueeze(0).expand(s_s.shape[0], -1)
        s_z_proj = torch.cat([
            s_z,
            self.seq_to_pair(s_s),
            self.positional_embedding(residx, mask=pair_mask),
        ], dim=-1)
        s_z_proj = self.projection(s_z_proj)
        return s_z_proj, pair_mask.to(s_z_proj), mask.to(s_z_proj)

    def _single_pass(self, s_z_proj, pair_mask, dists):
        """One recycling pass (for early-stop path)."""
        s_z_c = s_z_proj + self.recycle(dists)
        s_z_c = self.miniformer(s_z_c, pair_mask)
        preds = self.fc_out(s_z_c + s_z_c.transpose(1, 2))
        dists = preds.detach().argmax(dim=-1)
        dists = nn.functional.one_hot(dists, self.disto_bins).to(s_z_proj)
        return preds, s_z_c, dists


class MiniFoldModel(nn.Module):
    def __init__(
        self,
        esm_model_name: str,
        num_blocks: int = 4,
        no_bins: int = 64,
        use_structure_module: bool = False,
        config_of: dict = None,
        num_structure_blocks: int = 8,
        kernels: bool = False,
    ):
        super().__init__()
        # Save args
        self.esm_model_name = esm_model_name
        self.num_blocks = num_blocks
        self.use_structure_module = use_structure_module
        self.kernels = kernels

        # Language model encoder
        lm, _ = load_model_and_alphabet(esm_model_name)
        lm.lm_head = nn.Identity()
        lm.contact_head = nn.Identity()
        embed_dim = lm.embed_dim
        attn_dim = lm.attention_heads * lm.num_layers
        self.lm = lm

        # Freeze all but the last 2 layers in the PLM
        for name, param in self.lm.named_parameters():
            if (
                ("layers.34" in name)
                or ("layers.35" in name)
                or ("emb_layer_norm_after" in name)
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Layers
        self.fc_s = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.fold = FoldingTrunk(
            c_s=1024,
            c_z=128,
            bins=32,
            disto_bins=no_bins,
            num_layers=num_blocks,
            kernels=kernels,
        )

        self.use_structure_module = use_structure_module
        if use_structure_module:
            self.sz_project = PairToSequence(c_z=128, c_s=1024)
            self.structure_module = StructureModule(
                c_s=1024,
                c_z=128,
                c_resnet=128,
                head_dim=64,
                no_heads=16,
                no_blocks=num_structure_blocks,
                no_resnet_blocks=2,
                no_angles=7,
                trans_scale_factor=10,
                epsilon=1e-5,
                inf=1e5,
            )
            self.aux_heads = AuxiliaryHeads(
                config_of.model["heads"],
            )

    def forward(self, batch: dict, num_recycling: int = 0, early_stop: float = None):
        # Check if kernels are on
        if self.training and self.kernels:
            raise ValueError("Kernels can only be used at inference.")

        # For some reason having inference issues with autocast on the LM
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        with torch.autocast(device.type, enabled=self.training, dtype=torch.bfloat16):
            idx = self.lm.num_layers
            lm_out = self.lm(batch["seq"], repr_layers=[idx], need_head_weights=True)
            s_s = lm_out.pop("representations")[idx]

        # Compute sequence embeddings
        s_s = self.fc_s(s_s)

        # Compute attention embeddings
        B, N = s_s.shape[:2]
        s_z = lm_out.pop("attentions")
        s_z = s_z.view(B, N, N, -1)
        s_z = self.fc_z(s_z)

        if early_stop is not None and self.use_structure_module:
            # Early-stop path: run structure module + pLDDT after each pass
            from minifold.train.loss import compute_plddt
            feats = tensor_tree_map(lambda t: t[..., 0], batch["batch_of"])
            s_z_proj, pair_mask_f, _ = self.fold._project(s_s, s_z, batch["mask"])
            shape = tuple(s_z_proj.shape[:3]) + (self.fold.disto_bins,)
            dists = torch.zeros(shape, device=s_z_proj.device, dtype=s_z_proj.dtype)

            preds, s_z_c, single, sm_out = None, None, None, None
            for i in range(num_recycling + 1):
                with torch.set_grad_enabled(self.training and (i == num_recycling)):
                    preds, s_z_c, dists = self.fold._single_pass(s_z_proj, pair_mask_f, dists)
                single = self.sz_project(s_z_c, s_s, pair_mask_f)
                sm_out = self.structure_module(
                    s=single, z=s_z_c,
                    aatype=feats["aatype"],
                    mask=feats["seq_mask"].to(dtype=single.dtype),
                )
                mean_plddt = float(compute_plddt(self.aux_heads.plddt(sm_out["single"])).mean())
                print(f"  pass {i+1}/{num_recycling+1}: pLDDT={mean_plddt:.1f}", flush=True)
                if mean_plddt >= early_stop and i < num_recycling:
                    print(f"  → early stop (>= {early_stop})", flush=True)
                    break

            r_dict = {"preds": preds, "pair": s_z_c, "single": single, "sm": sm_out}
            r_dict["final_atom_positions"] = atom14_to_atom37(sm_out["positions"][-1], feats)
            r_dict["final_atom_mask"] = feats["atom37_atom_exists"]
            r_dict["final_affine_tensor"] = sm_out["frames"][-1]
            r_dict.update(self.aux_heads(r_dict))
            return r_dict

        # Standard path
        preds, s_z = self.fold(
            s_s,
            s_z,
            mask=batch["mask"],
            num_recycling=num_recycling,
        )

        # Prepare output
        r_dict = {"preds": preds, "pair": s_z}

        # Run structure module
        if self.use_structure_module:
            mask = batch["mask"]
            pair_mask = mask[:, None, :] * mask[:, :, None]
            r_dict["single"] = self.sz_project(s_z, s_s, pair_mask)

            # No recycle differences in feats
            feats = tensor_tree_map(lambda t: t[..., 0], batch["batch_of"])

            r_dict["sm"] = self.structure_module(
                s=r_dict["single"],
                z=r_dict["pair"],
                aatype=feats["aatype"],
                mask=feats["seq_mask"].to(dtype=r_dict["single"].dtype),
            )

            r_dict["final_atom_positions"] = atom14_to_atom37(
                r_dict["sm"]["positions"][-1], feats
            )
            r_dict["final_atom_mask"] = feats["atom37_atom_exists"]
            r_dict["final_affine_tensor"] = r_dict["sm"]["frames"][-1]

            # Run auxiliary heads
            r_dict.update(self.aux_heads(r_dict))

        return r_dict
