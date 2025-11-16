import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        # Initialize the inherited class, nn.linear 
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout) 

            self.lora_scaling = lora_alpha / lora_rank 

            factory_kwargs = {"device": device, "dtype": dtype}
            self.lora_A = nn.parameter.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.parameter.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            # Initialize both lora_A and lora_B
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Make sure to merge in LORA matrices only if not already merged 
        if self.is_lora() and not self.has_weights_merged:
            lora_weight = self.lora_B @ self.lora_A
            return F.linear(input=input, weight=self.weight, bias=self.bias) + self.lora_scaling * self.lora_dropout(F.linear(input=input, weight=lora_weight, bias=torch.zeros_like(self.bias)))
        else:
            return F.linear(input=input, weight=self.weight, bias=self.bias)

    def train(self, mode: bool = True) -> "LoRALinear":
        # Set the linear layer into train mode
        super().train(mode=mode)
        if self.is_lora():
            if self.has_weights_merged:
                self.weight.data -= self.lora_scaling * self.lora_B @ self.lora_A
                self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        # Set the linear layer into eval mode
        super().eval()
        if self.is_lora():
            if not self.has_weights_merged:
                self.weight.data += self.lora_scaling * self.lora_B @ self.lora_A
                self.has_weights_merged = True
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out
    

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    for param_name, param in model.named_parameters():
        if "lora" in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def mark_initial_weight_as_not_trainable(model: nn.Module, initial_param_name: str) -> nn.Module:
    for param_name, param in model.named_parameters():
        if initial_param_name in param_name and "lora" not in param_name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


def mark_param_not_in_transformer_as_trainable(model: nn.Module) -> nn.Module:
    for param_name, param in model.named_parameters():
        if "transformer" not in param_name:
            param.requires_grad = True

    return model


def replace_linear_with_lora_in_mha_in_proj(mha: nn.MultiheadAttention,
                                   lora_rank=4,
                                   lora_alpha=1.0,
                                   lora_dropout=0.0):
    d_model = mha.embed_dim  
    w_qkv = mha.in_proj_weight  
    b_qkv = mha.in_proj_bias   

    q_weight, k_weight, v_weight = w_qkv.chunk(3, dim=0)
    if b_qkv is not None:
        q_bias, k_bias, v_bias = b_qkv.chunk(3)
    else:
        q_bias = k_bias = v_bias = None

    in_proj_q = LoRALinear(
            in_features=d_model, out_features=d_model, bias=(q_bias is not None),
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=w_qkv.device, dtype=w_qkv.dtype
        )
    in_proj_k = LoRALinear(
            in_features=d_model, out_features=d_model, bias=(k_bias is not None),
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=w_qkv.device, dtype=w_qkv.dtype
        )
    in_proj_v = LoRALinear(
            in_features=d_model, out_features=d_model, bias=(v_bias is not None),
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=w_qkv.device, dtype=w_qkv.dtype
        )


    with torch.no_grad():
        in_proj_q.weight.copy_(q_weight)
        if q_bias is not None:
            in_proj_q.bias.copy_(q_bias)

        in_proj_k.weight.copy_(k_weight)
        if k_bias is not None:
            in_proj_k.bias.copy_(k_bias)

        in_proj_v.weight.copy_(v_weight)
        if v_bias is not None:
            in_proj_v.bias.copy_(v_bias)

    mha.in_proj_q = in_proj_q
    mha.in_proj_k = in_proj_k
    mha.in_proj_v = in_proj_v

    mha.in_proj_weight = None
    mha.in_proj_bias = None

    original_forward = mha.forward

    def lora_mha_forward(query, key, value, **kwargs):
        attn_mask = kwargs.get("attn_mask", None)
        need_weights = kwargs.get("need_weights", False)
        key_padding_mask = kwargs.get("key_padding_mask", None)

        Q = mha.in_proj_q(query)
        K = mha.in_proj_k(key)
        V = mha.in_proj_v(value)

        return torch.nn.functional.multi_head_attention_forward(
            Q, K, V,
            embed_dim_to_check=mha.embed_dim,
            num_heads=mha.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=mha.bias_k,
            bias_v=mha.bias_v,
            add_zero_attn=mha.add_zero_attn,
            dropout_p=mha.dropout,
            out_proj_weight=mha.out_proj.weight,
            out_proj_bias=mha.out_proj.bias,
            training=mha.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,  
            q_proj_weight=mha.in_proj_q.weight,
            k_proj_weight=mha.in_proj_k.weight,
            v_proj_weight=mha.in_proj_v.weight,
        )

    mha.forward = lora_mha_forward


def replace_linear_with_lora_mha_inproj(module: nn.Module, lora_rank=4, lora_alpha=1.0, lora_dropout=0.0):
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            replace_linear_with_lora_in_mha_in_proj(child, lora_rank, lora_alpha, lora_dropout)
        else:
            replace_linear_with_lora_mha_inproj(child, lora_rank, lora_alpha, lora_dropout)


def replace_linear_with_lora_mha_outproj(
    module: nn.Module,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0
):
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            if isinstance(child.out_proj, nn.Linear):
                old_linear = child.out_proj
                new_lora = LoRALinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        bias=(old_linear.bias is not None),
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        device=old_linear.weight.device,
                        dtype=old_linear.weight.dtype
                )
                new_lora.weight.data = old_linear.weight.data.clone()
                if old_linear.bias is not None:
                    new_lora.bias.data = old_linear.bias.data.clone()

                child.out_proj = new_lora

        else:
            replace_linear_with_lora_mha_outproj(child, lora_rank, lora_alpha, lora_dropout)