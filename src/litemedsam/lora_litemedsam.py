from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from src.litemedsam.build_sam import MedSAM_Lite
from safetensors import safe_open
from safetensors.torch import save_file


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # print(qkv.shape), print(new_q.shape), print(new_v.shape)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class LoRA_liteMedSam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        litemedsam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA. [[0,1],[0,1],[0,1,2,3,4,5],[0,1]]

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, litemedsam_model: MedSAM_Lite, r: int, lora_layer=None):
        super(LoRA_liteMedSam, self).__init__()

        assert r > 0
        # base_vit_dim = litemedsam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # self.lora_layer = list(range(len(litemedsam_model.image_encoder.layers)))
            self.lora_layer = [[0,1],[0,1],[0,1,2,3,4,5],[0,1]]
        # create for storage, then we can init them or load weights
        self.A_weights = []  # These are linear layers
        self.B_weights = []

        # lets freeze first
        for param in litemedsam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for i, layer in enumerate(litemedsam_model.image_encoder.layers):
            # for swin structure litemedsam encoder
            if i != 0:
                for t_layer_i, blk in enumerate(layer.blocks):
                    # If we only want few lora layer instead of all
                    if t_layer_i not in self.lora_layer[i]:
                        continue
                    w_qkv_linear = blk.attn.qkv
                    self.dim = w_qkv_linear.in_features
                    w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                    w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                    self.A_weights.append(w_a_linear_q)
                    self.B_weights.append(w_b_linear_q)
                    self.A_weights.append(w_a_linear_v)
                    self.B_weights.append(w_b_linear_v)
                    blk.attn.qkv = _LoRA_qkv(
                        w_qkv_linear,
                        w_a_linear_q,
                        w_b_linear_q,
                        w_a_linear_v,
                        w_b_linear_v,
                    )
        self.reset_parameters()
        self.litemedsam = litemedsam_model

    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_liteMedSam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)


# if __name__ == "__main__":
#     sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
#     LoRA_liteMedSam = LoRA_liteMedSam(sam,4)
#     LoRA_liteMedSam.sam.image_encoder(torch.rand(size=(1,3,1024,1024)))