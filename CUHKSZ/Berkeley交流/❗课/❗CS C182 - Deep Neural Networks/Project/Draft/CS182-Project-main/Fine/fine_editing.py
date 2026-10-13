# fine_editing.py
# FiNE: Precise Localization of Memories (ICLR 2025)
# - 4.1 Locating Neurons in LLMs  (Eq. (3), Algorithm 1)
# - 4.2 Updating Knowledge       (Eq. (4)(5)(6)(7))
# 适配 GPT-NeoX / Pythia / LLaMA / GPT-J 风格，并兼容 8bit + device_map="auto" 的加载方式。

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_lm_head(model: nn.Module) -> nn.Linear:
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head
    raise AttributeError(
        "Model has no `lm_head` or `embed_out`; cannot obtain W_u for FiNE."
    )

# ========= 基本数据结构 =========

@dataclass
class EditSpec:
    """单条编辑任务，对应论文里的 (s_j, r_j, o_j, o*_j)."""
    prompt_edit: str   # p(s_j, r_j)
    old_answer: str    # o_j
    new_answer: str    # o*_j


@dataclass
class FineConfig:
    """
    FiNE 的主要超参数（4.2 + 4.3）

    lf:   layer freezing 截断数，最后 lf 层不参与（论文 4.3）
    k:    Algorithm 1 里每个 token 选的 top-k neuron 数
    epochs: 优化 Z_j 的最大迭代步数
    lr:   优化 Z_j 的学习率
    alpha_kl, beta_pen: Eq. (7) 的 α, β
    gamma_l2: 对 Z_j 的 L2 正则
    early_stop_prob: 当目标 token 概率足够高时提前结束（在 exp(-loss_edit) 上阈值）
    device: 主要用于放输入 tensor；模型本身不再强制 .to(device)
    """
    lf: int = 4
    k: int = 20

    epochs: int = 50
    lr: float = 1e-2
    alpha_kl: float = 1.0
    beta_pen: float = 0.1
    gamma_l2: float = 1e-4
    early_stop_prob: float = 0.95

    device: str = "cuda"


# ========= 工具：模型结构适配 =========

def _get_num_layers(model: nn.Module) -> int:
    """返回 Transformer block 的层数."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise AttributeError(
        "Cannot infer number of layers; expected gpt_neox.layers / model.layers / transformer.h."
    )


def _get_block(model: nn.Module, layer_id: int) -> nn.Module:
    """取第 layer_id 个 Transformer block."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_id]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_id]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_id]
    raise AttributeError("Cannot get transformer block; unsupported architecture.")


def _get_ffn_out_layer(model: nn.Module, layer_id: int) -> nn.Linear:
    """
    返回第 layer_id 层 FFN 的“第二个线性层” W_out^l.

    兼容：
      - GPT-NeoX / Pythia: block.mlp.dense_4h_to_h
      - LLaMA:             block.mlp.down_proj
      - GPT-J / GPT-2:     block.mlp.fc_out / block.mlp.c_proj
    """
    block = _get_block(model, layer_id)
    if hasattr(block, "mlp"):
        mlp = block.mlp
    elif hasattr(block, "ff"):
        mlp = block.ff
    else:
        raise AttributeError(f"Layer {layer_id} has no 'mlp' or 'ff'.")

    # GPT-NeoX / Pythia
    if hasattr(mlp, "dense_4h_to_h"):
        return mlp.dense_4h_to_h

    # LLaMA
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


# ========= 4.1: Eq.(3) + Algorithm 1 – neuron localization =========

@torch.no_grad()
def locate_neurons_fine(
    model: nn.Module,
    tokenizer,
    edit_spec: EditSpec,
    layers: List[int],
    k: int,
) -> List[Tuple[int, int]]:
    """
    根据 FiNE 论文 Section 4.1 的 Eq.(3) 与 Algorithm 1，
    定位单条 (s_j, r_j, o_j, o*_j) 对应的 neuron 集合 U_j。

    我们实现的评分：
        c(i, l, t) = | q_l(t, i) * (W_u W_out^l)_{t, i} |
    并对每个 token t ∈ o_j 取 top-k，再对所有 t 做并集。
    """
    device = next(model.parameters()).device

    # 1) 前向，记录 FFN 输出前激活 q_l(t)
    prompt = edit_spec.prompt_edit
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    q_store: Dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(module, inp, out):
            # out: [B, T, d_m]，我们假定 B=1
            q_store[layer_id] = out.detach().squeeze(0)  # [T, d_m]
        return hook

    handles = []
    for l in layers:
        ffn_out = _get_ffn_out_layer(model, l)
        h = ffn_out.register_forward_hook(make_hook(l))
        handles.append(h)

    _ = model(**inputs)

    for h in handles:
        h.remove()

    # 2) 准备 W_u (lm_head.weight) 与 object tokens
    lm_head = _get_lm_head(model)
    Wu = lm_head.weight.data  # [vocab, d_h]

    obj_ids = tokenizer(edit_spec.old_answer, add_special_tokens=False)["input_ids"]

    neuron_set: set[Tuple[int, int]] = set()

    for token_id in obj_ids:
        token_id_t = torch.tensor([token_id], device=device, dtype=torch.long)
        contrib_scores: Dict[Tuple[int, int], float] = {}

        for l in layers:
            if l not in q_store:
                continue
            q_all = q_store[l]             # [T, d_m]
            q_l = q_all.mean(dim=0)        # 简单平均，得到一个 “代表性” q_l ∈ R^{d_m}

            ffn_out = _get_ffn_out_layer(model, l)
            W_out = ffn_out.weight.data    # [d_h, d_m]（假定 out_features=d_h, in_features=d_m）

            Wu_t = Wu[token_id_t, :]       # [1, d_h]
            proj = Wu_t @ W_out            # [1, d_m]
            proj = proj.squeeze(0)         # [d_m]

            c_vec = (q_l * proj).abs().cpu()   # [d_m]

            for i in range(c_vec.shape[0]):
                contrib_scores[(l, i)] = float(c_vec[i].item())

        # 对该 token 的所有 (l,i) 做 top-k
        sorted_neurons = sorted(contrib_scores.items(), key=lambda x: x[1], reverse=True)
        top_t = sorted_neurons[:k]
        for (layer_id, neuron_idx), _score in top_t:
            neuron_set.add((layer_id, neuron_idx))

    U_j = sorted(list(neuron_set))
    return U_j


# ========= 4.2: Z_j 参数 + Forward Hook 实现 ΔW_out =========

class FinePatch(nn.Module):
    """
    FiNE 的 Z_j 参数容器 + forward hook 的实现。

    对每个 neuron u = (layer_id, neuron_idx) 分配向量 z_u ∈ R^{d_in}，
    并在 FFN 输出层上实现：
        W_out^l'[i,:] = W_out^l[i,:] + z_u

    在 forward hook 中等价实现：
        y = x @ W_out^T   # 原输出
        y_i += (x * z_u).sum(-1)
    """

    def __init__(
        self,
        model: nn.Module,
        neurons: List[Tuple[int, int]],
        device: str,
    ):
        super().__init__()
        self.model = model
        self.device = device

        layer2idx: Dict[int, List[int]] = {}
        for (l, i) in neurons:
            layer2idx.setdefault(l, []).append(i)
        self.layer2idx = layer2idx

        self.z_params: Dict[Tuple[int, int], nn.Parameter] = {}
        self.z_list = nn.ParameterList()
        self.handles: List[Any] = []

        for l, idx_list in layer2idx.items():
            ffn_out = _get_ffn_out_layer(model, l)
            d_in = ffn_out.in_features

            z_this_layer: Dict[int, nn.Parameter] = {}
            for i in idx_list:
                z = nn.Parameter(torch.zeros(d_in, device=device, dtype=torch.float32))
                self.z_list.append(z)
                z_this_layer[i] = z
                self.z_params[(l, i)] = z

            handle = ffn_out.register_forward_hook(
                self._make_layer_hook(idx_list, z_this_layer)
            )
            self.handles.append(handle)

    def _make_layer_hook(self, neuron_indices: List[int], z_this_layer: Dict[int, nn.Parameter]):
        def hook(module, inp, out):
            # inp[0]: x ∈ R^{B,T,d_in}
            x = inp[0]
            for i in neuron_indices:
                if i >= out.shape[-1]:
                    continue
                z = z_this_layer[i]  # [d_in]
                delta = (x * z).sum(dim=-1)  # [B,T]
                out[..., i] = out[..., i] + delta
            return out
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def parameters(self, recurse: bool = True):
        return self.z_list.parameters(recurse=recurse)


# ========= 构造编辑 batch （Eq.(4)(5)） =========

def _build_edit_batch(
    tokenizer,
    prompt_edit: str,
    old_answer: str,
    new_answer: str,
    device: torch.device,
):
    """
    构造：
      - batch_new: p(s,r) + o*_j （用于 L_edit）
      - batch_old: p(s,r) + o_j  （用于 L_old 与 KL）
    """
    enc_prompt = tokenizer(prompt_edit, return_tensors="pt", add_special_tokens=False)
    enc_old = tokenizer(old_answer, return_tensors="pt", add_special_tokens=False)
    enc_new = tokenizer(new_answer, return_tensors="pt", add_special_tokens=False)

    def _concat(enc_ans):
        input_ids = torch.cat(
            [enc_prompt["input_ids"], enc_ans["input_ids"]], dim=1
        )
        attention_mask = torch.cat(
            [enc_prompt["attention_mask"], enc_ans["attention_mask"]], dim=1
        )
        labels = input_ids.clone()
        prefix_len = enc_prompt["input_ids"].shape[1]
        labels[:, :prefix_len] = -100
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    batch_new = _concat(enc_new)
    batch_old = _concat(enc_old)
    return batch_new, batch_old


# ========= 4.2: 损失计算（Eq.(4)(5)(6)(7)） =========

def fine_loss_step(
    model: nn.Module,
    patch: FinePatch,
    batch_new: Dict[str, torch.Tensor],
    batch_old: Dict[str, torch.Tensor],
    logits_base: torch.Tensor,
    logp_old_base: torch.Tensor,
    cfg: FineConfig,
):
    """
    L_edit   = - log P_{M'}(o*_j | p)                (Eq. (4))
    L_old    = - log P_{M'}(o_j   | p)               (Eq. (5))
    L_pen    = max(0, logP_M(o_j|p) - logP_{M'}(o_j|p))  (Eq. (6))
    L_total  = L_edit + α·KL(P_M || P_{M'}) + β·L_pen + γ‖Z_j‖²  (Eq. (7))
    """
    device = next(model.parameters()).device

    # 1) L_edit（new answer）
    outputs_new = model(**batch_new)
    loss_edit = outputs_new.loss  # mean NLL over new answer tokens

    # 2) L_old + KL （old answer under edited model）
    outputs_old_edit = model(**batch_old)
    logits_edit = outputs_old_edit.logits  # [1, T, V]
    loss_old_edit = outputs_old_edit.loss

    labels = batch_old["labels"]
    mask = labels != -100

    log_probs_base = F.log_softmax(logits_base, dim=-1)
    log_probs_edit = F.log_softmax(logits_edit, dim=-1)

    p_base = log_probs_base.exp()
    kl_token = (p_base * (log_probs_base - log_probs_edit)).sum(-1)  # [1,T]
    kl_masked = kl_token[mask]
    kl_mean = kl_masked.mean()

    logp_old_edit = -loss_old_edit
    diff = logp_old_base - logp_old_edit
    loss_pen = torch.clamp(diff, min=0.0)

    # ‖Z_j‖²
    l2_reg = torch.tensor(0.0, device=device)
    for z in patch.parameters():
        l2_reg = l2_reg + (z ** 2).sum()

    loss_total = loss_edit + cfg.alpha_kl * kl_mean + cfg.beta_pen * loss_pen + cfg.gamma_l2 * l2_reg

    stats = {
        "loss_edit": float(loss_edit.detach().cpu().item()),
        "loss_kl": float(kl_mean.detach().cpu().item()),
        "loss_pen": float(loss_pen.detach().cpu().item()),
        "l2": float(l2_reg.detach().cpu().item()),
        "logp_old_base": float(logp_old_base.detach().cpu().item()),
        "logp_old_edit": float(logp_old_edit.detach().cpu().item()),
    }
    return loss_total, stats


# ========= 4.2 + 4.3: 主入口 =========

def apply_fine_edit(
    model: nn.Module,
    tokenizer,
    edit_spec: EditSpec,
    fine_config: Dict[str, Any],
):
    """
    主入口：对单条 (s_j, r_j, o_j, o*_j) 执行 FiNE 编辑。

    注意：
      - 不再对模型调用 .to(cfg.device)，直接使用现有参数所在 device，
        以兼容 8bit + device_map="auto" 的加载方式。
    """
    cfg = FineConfig(**fine_config) if isinstance(fine_config, dict) else fine_config

    device = next(model.parameters()).device
    cfg.device = str(device)
    model.eval()

    # 1) layer freezing: 只在前 L - lf 层里找 neuron
    num_layers = _get_num_layers(model)
    max_layer_for_loc = max(0, num_layers - cfg.lf)
    layers = list(range(max_layer_for_loc))

    # 2) neuron localization（Eq.(3) + Algorithm 1）
    neurons = locate_neurons_fine(
        model=model,
        tokenizer=tokenizer,
        edit_spec=edit_spec,
        layers=layers,
        k=cfg.k,
    )

    if not neurons:
        return model, {
            "method": "FiNE",
            "edited": False,
            "reason": "no_neuron_found",
            "layers_used": layers,
        }

    # 3) 构造编辑 batch
    batch_new, batch_old = _build_edit_batch(
        tokenizer,
        edit_spec.prompt_edit,
        edit_spec.old_answer,
        edit_spec.new_answer,
        device=device,
    )

    # baseline：P_M(o_j|p) & logits_base（不带 patch）
    with torch.no_grad():
        outputs_old_base = model(**batch_old)
        logits_base = outputs_old_base.logits.detach()
        loss_old_base = outputs_old_base.loss.detach()
        logp_old_base = -loss_old_base

    # 4) 创建 Z_j (FinePatch) 并优化
    patch = FinePatch(model=model, neurons=neurons, device=str(device))
    optimizer = torch.optim.Adam(patch.parameters(), lr=cfg.lr)

    best_loss = float("inf")
    best_state = None
    best_prob_new = 0.0

    for step in range(cfg.epochs):
        optimizer.zero_grad()
        loss, stats = fine_loss_step(
            model=model,
            patch=patch,
            batch_new=batch_new,
            batch_old=batch_old,
            logits_base=logits_base,
            logp_old_base=logp_old_base,
            cfg=cfg,
        )
        loss.backward()
        optimizer.step()

        # 目标 token 的平均 logP
        avg_logp_new = -stats["loss_edit"]
        prob_new = float(torch.exp(torch.tensor(avg_logp_new)))

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in patch.state_dict().items()}

        if prob_new > cfg.early_stop_prob and prob_new > best_prob_new:
            best_prob_new = prob_new
            break

    if best_state is not None:
        patch.load_state_dict(best_state)

    info = {
        "method": "FiNE",
        "edited": True,
        "neurons": neurons,
        "num_neurons": len(neurons),
        "config": cfg.__dict__,
        "best_loss": best_loss,
    }

    # 把 patch 挂在 model 上，之后直接用该 model 推理即可
    model.fine_patch = patch
    return model, info
