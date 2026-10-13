# editing_methods.py
# -----------------------------------------------------------
# Editing methods used in experiments:
#   - FiNE: neuron-level editing (Precise Localization of Memories)
#   - ROME: placeholder / baseline interface
#
# 本文件的职责：
#   1. 提供 compute_fine_contributions 给 overlap_dataset.py 用，
#      按 FiNE Eq.(3) 计算 neuron 贡献分数；
#   2. 暴露 EditSpec / apply_fine_edit / apply_rome_edit 给 run_experiment.py 用；
#      其中 FiNE 的真正实现放在 fine_editing.py 里，这里只转发。
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn

# 从 fine_editing 中导入完整的 FiNE 实现
from fine_editing import apply_fine_edit as fine_apply_fine_edit
from fine_editing import EditSpec, FineConfig


# ================== 工具：取得 FFN 第二层 W_out ==================

def _get_block(model: nn.Module, layer_id: int) -> nn.Module:
    """取第 layer_id 个 Transformer block，兼容 GPT-NeoX / LLaMA / GPT-J 风格。"""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_id]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_id]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_id]
    raise AttributeError(
        "Unsupported architecture: expected gpt_neox.layers / model.layers / transformer.h"
    )


def _get_ffn_out_layer(model: nn.Module, layer_id: int) -> nn.Linear:
    """
    返回第 layer_id 层 FFN 的第二个线性层 W_out^l，
    用于 FiNE Eq.(3) 里的 W_out^l。
    """
    block = _get_block(model, layer_id)
    if hasattr(block, "mlp"):
        mlp = block.mlp
    elif hasattr(block, "ff"):
        mlp = block.ff
    else:
        raise AttributeError(f"Layer {layer_id} has no 'mlp' or 'ff' attribute.")

    # GPT-NeoX / Pythia
    if hasattr(mlp, "dense_4h_to_h"):
        return mlp.dense_4h_to_h
    # LLaMA / Mistral
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    # GPT-J / GPT-2
    if hasattr(mlp, "fc_out"):
        return mlp.fc_out
    if hasattr(mlp, "c_proj"):
        return mlp.c_proj

    raise AttributeError(
        f"Cannot find FFN output linear in layer {layer_id}; "
        f"tried dense_4h_to_h / down_proj / fc_out / c_proj."
    )

def _get_lm_head(model: nn.Module) -> nn.Linear:
    """
    返回用于生成 token logits 的输出层（即 W_u）。
    兼容 GPT-NeoX / Pythia / LLaMA / GPT-J 等。
    """
    # 常见：GPT-2 / GPT-J / 部分模型
    if hasattr(model, "lm_head"):
        return model.lm_head

    # Pythia / GPT-NeoX：通常叫 embed_out
    if hasattr(model, "embed_out"):
        return model.embed_out

    # 有些模型把 head 放在 model 里面
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head

    raise AttributeError(
        "Model has no `lm_head` or `embed_out`; cannot obtain W_u for FiNE Eq.(3)."
    )

# ================== FiNE Eq.(3)：neuron contribution ==================

@torch.no_grad()
def compute_fine_contributions(
    model: nn.Module,
    tokenizer,
    fact_prompt: str,
    target_answer: str = None,
    object_text: str = None,
    layers: List[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    实现 FiNE 论文 Section 4.1 的 Eq.(3) 的打分部分：

        c(i, l, t) = q^l_{i,-1} * (W_u W_out^l)_{t, i}

    兼容两种调用方式：
      - compute_fine_contributions(..., target_answer=..., layers=...)
      - compute_fine_contributions(..., object_text=..., layers=...)

    最终用 object_text 作为 o_j，取其第一个 token 作为目标 token t。
    """
    if layers is None:
        raise ValueError("layers must be provided")

    # 兼容老参数名：如果 object_text 没给，就用 target_answer
    if object_text is None:
        if target_answer is None:
            raise ValueError("Either object_text or target_answer must be provided")
        object_text = target_answer

    device = next(model.parameters()).device

    # 1) 准备 (W_u, object token)
    lm_head = _get_lm_head(model)
    Wu = lm_head.weight.data  # [vocab, d_h]

    obj_ids = tokenizer(object_text, add_special_tokens=False)["input_ids"]
    if len(obj_ids) == 0:
        return {}
    t_id = obj_ids[0]  # 目标 token t

    # 2) 前向，拦截 FFN 第二层输入（即 q^l）
    inputs = tokenizer(fact_prompt, return_tensors="pt").to(device)

    q_store: Dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(mod, inp, out):
            # inp[0]: [B, T, d_m]，我们取最后一个 token 的 q^l_{·,-1}
            q_store[layer_id] = inp[0][0, -1, :].detach()  # [d_m]
        return hook

    handles = []
    for l in layers:
        ffn_out = _get_ffn_out_layer(model, l)
        h = ffn_out.register_forward_hook(make_hook(l))
        handles.append(h)

    _ = model(**inputs)

    for h in handles:
        h.remove()

    # 3) 计算每个 (l,i) 的贡献分数
    neuron_scores: Dict[Tuple[int, int], float] = {}

    t_id_tensor = torch.tensor([t_id], device=device, dtype=torch.long)
    Wu_t = Wu[t_id_tensor, :]  # [1, d_h]

    for l in layers:
        if l not in q_store:
            continue

        q_l = q_store[l]              # [d_m]
        ffn_out = _get_ffn_out_layer(model, l)
        W_out = ffn_out.weight.data.to(torch.float32)  # ← 转 fp32
        Wu_t = Wu_t.to(torch.float32)  # ← 也转一致
        proj = (Wu_t @ W_out).squeeze(0)  # OK

        contrib_vec = (q_l * proj).abs().cpu()  # [d_m]
        for i in range(contrib_vec.shape[0]):
            neuron_scores[(l, i)] = float(contrib_vec[i].item())

    return neuron_scores


# ================== FiNE 编辑入口（转发到 fine_editing） ==================

def apply_fine_edit(
    model: nn.Module,
    tokenizer,
    edit_spec: EditSpec,
    fine_config: Dict[str, Any],
):
    """
    包一层，直接调用 fine_editing.py 里的完整 FiNE 实现。

    fine_config 应该包含：
      - lf, k, epochs, lr, alpha_kl, beta_pen, gamma_l2, early_stop_prob, device
    """
    # 这里直接复用 FineConfig / fine_apply_fine_edit，内部已经完全按 FiNE Eq.(4)(5)(6)(7) 实现。
    if not isinstance(fine_config, FineConfig):
        cfg = FineConfig(**fine_config)
    else:
        cfg = fine_config

    return fine_apply_fine_edit(model, tokenizer, edit_spec, cfg.__dict__)


# ================== ROME 编辑（占位接口，可改成真实实现） ==================

def apply_rome_edit(
    model: nn.Module,
    tokenizer,
    edit_spec: EditSpec,
    rome_config: Dict[str, Any],
):
    """
    ROME 的占位实现，用来对比 FiNE。
    目前只是“啥也不改”，直接返回原模型和一些信息。

    你之后可以用真实 ROME 代码替换这里：
      - 从 `easy-knowledge-editing` / `rome` 项目里拷主函数
      - 或做一个简单的线性层 patch 做 baseline
    """
    info = {
        "method": "ROME",
        "edited": False,
        "reason": "rome_not_implemented_placeholder",
        "config": rome_config,
    }
    return model, info
