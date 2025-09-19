from segment_anything.modeling.sam import Sam

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file
import yaml


class LoRA_qkv(nn.Module):
    """Low-rank adaptor that augments SAM attention qkv projections.

    Args:
        qkv (nn.Linear): Baseline projection producing concatenated query/key/value vectors.
        linear_a_q (nn.Module): LoRA A matrix for the query branch.
        linear_b_q (nn.Module): LoRA B matrix for the query branch.
        linear_a_v (nn.Module): LoRA A matrix for the value branch.
        linear_b_v (nn.Module): LoRA B matrix for the value branch.
        lora_dropout (float): Dropout probability applied prior to each LoRA A projection.
        lora_alpha (float): Scaling factor; effective delta equals ``alpha / rank`` times the LoRA output.
        rank (int): LoRA rank used for scaling.

    Shapes:
        x: ``(B, N, d)``
        output: ``(B, N, 3 * d)`` with deltas applied to query/value slices.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        lora_dropout: float = 0.1,
        lora_alpha: float = 1.0,
        rank: int = 1,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.d_model = qkv.in_features
        self.out_features = qkv.out_features
        assert self.out_features == 3 * self.d_model, (
            f"qkv.out_features={self.out_features} but expected 3*d where d={self.d_model}"
        )

        self.drop_q = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()
        self.drop_v = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

        # Scale LoRA delta to keep update magnitude stable
        self.scaling = float(lora_alpha) / float(max(rank, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the LoRA-augmented projection to the input sequence."""
        qkv = self.qkv(x)  # [B, N, 3*d]

        # LoRA branches (Q and V)
        x_q = self.drop_q(x)
        x_v = self.drop_v(x)

        q_ba = self.linear_b_q(self.linear_a_q(x_q)) * self.scaling  # [B, N, d]
        v_ba = self.linear_b_v(self.linear_a_v(x_v)) * self.scaling  # [B, N, d]

        # Apply deltas to query (leading d) and value (trailing d) slices only.
        # q: first d, k: middle d, v: last d
        qkv[..., :self.d_model] = qkv[..., :self.d_model] + q_ba
        qkv[..., -self.d_model:] = qkv[..., -self.d_model:] + v_ba

        return qkv


class LoRA_sam(nn.Module):
    """Wrap SAM image-encoder blocks with LoRA adapters on query/value projections.

    Args:
        sam_model (Sam): Base SAM model to augment.
        rank (int): LoRA rank (>0).
        lora_layer (Iterable[int], optional): Indices of encoder blocks to adapt. Defaults to all blocks.
        lora_dropout (float): Dropout probability applied before each LoRA A projection.
        lora_alpha (float): Scaling factor controlling LoRA delta magnitude.

    Attributes:
        lora_layer (list[int]): Encoder block indices receiving LoRA updates.
        sam (Sam): Underlying SAM model instance.
        lora_vit (nn.Module): SAM image encoder with LoRA-wrapped attention.
    """

    def __init__(
        self,
        sam_model: Sam,
        rank: int,
        lora_layer=None,
        lora_dropout: float = 0.1,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        assert rank > 0, "LoRA rank must be > 0"
        self.rank = rank
        self.lora_dropout = float(lora_dropout)
        self.lora_alpha = float(lora_alpha)

        if lora_layer is not None:
            self.lora_layer = list(lora_layer)
            print(f"LoRA on layers: {self.lora_layer}")
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.A_weights = nn.ModuleList()
        self.B_weights = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for layer_idx, blk in enumerate(sam_model.image_encoder.blocks):
            if layer_idx not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            d_model = w_qkv_linear.in_features

            w_a_q = nn.Linear(d_model, self.rank, bias=False)
            w_b_q = nn.Linear(self.rank, d_model, bias=False)
            w_a_v = nn.Linear(d_model, self.rank, bias=False)
            w_b_v = nn.Linear(self.rank, d_model, bias=False)

            self.A_weights.append(w_a_q)
            self.B_weights.append(w_b_q)
            self.A_weights.append(w_a_v)
            self.B_weights.append(w_b_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_q, w_b_q,
                w_a_v, w_b_v,
                lora_dropout=self.lora_dropout,
                lora_alpha=self.lora_alpha,
                rank=self.rank,
            )

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder

    def reset_parameters(self):
        """Initialise LoRA weights with Kaiming A matrices and zero B matrices."""
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

    @torch.no_grad()
    def save_lora_parameters(self, filename: str):
        """Serialize LoRA weights (A/B) as safetensors for checkpointing.

        Args:
            filename (str): Destination file path.
        """
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(len(self.A_weights))}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(len(self.B_weights))}
        merged = {**a_tensors, **b_tensors}
        save_file(merged, filename)

    @torch.no_grad()
    def load_lora_parameters(self, filename: str):
        """Load LoRA weights produced by :meth:`save_lora_parameters`.

        Args:
            filename (str): Source safetensors path.
        """
        with safe_open(filename, framework="pt") as f:
            for i in range(len(self.A_weights)):
                tensor = f.get_tensor(f"w_a_{i:03d}")
                self.A_weights[i].weight.copy_(tensor)
            for i in range(len(self.B_weights)):
                tensor = f.get_tensor(f"w_b_{i:03d}")
                self.B_weights[i].weight.copy_(tensor)
