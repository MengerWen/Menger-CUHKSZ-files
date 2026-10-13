# ---------------------------
# FiNE: Neuron-Level Editing
# ---------------------------

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Tuple


# ======== 工具 1：计算 token 的平均激活（需要 forward hook） ========
def collect_hidden_states(model, tokenizer, prompt, layers):
    """
    Return: dict: layer -> hidden_states [seq_len, hidden_dim]
    """
    device = next(model.parameters()).device
    hs = {}

    def make_hook(layer_id):
        def hook(module, inp, out):
            hs[layer_id] = out.detach()
        return hook

    handles = []
    for layer_id in layers:
        h = model.transformer.h[layer_id].register_forward_hook(make_hook(layer_id))
        handles.append(h)

    # Forward
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model(**inputs)

    for h in handles:
        h.remove()

    return hs


# ======== 工具 2：算 neuron contribution（FiNE 原理简化版） ========
def compute_fine_contributions(model, tokenizer, fact_prompt, target_answer, layers):
    """
    返回: neuron_scores[(layer, neuron_id)] = importance score (float)
    用 simple "zero-out neuron" causal tracing 方法。
    """
    device = next(model.parameters()).device

    # baseline logprob(target)
    def get_logp(prompt, ans):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        ans_id = tokenizer(ans, add_special_tokens=False)["input_ids"][0]
        return logprobs[0, ans_id].item()

    baseline_lp = get_logp(fact_prompt, target_answer)

    neuron_scores = {}

    # get original hidden states
    hs = collect_hidden_states(model, tokenizer, fact_prompt, layers)

    # 对每个层，每个 neuron 做: zero-out，然后比较 logp 差异
    for layer_id in layers:
        h = hs[layer_id].detach().clone()  # [seq_len, hidden_dim]
        seq_len, hid = h.shape

        for neuron_id in range(hid):
            # 替换该 neuron 的 activation
            def hook_fn(mod, inp, out):
                out2 = out.clone()
                out2[:, :, neuron_id] = 0  # zero-out
                return out2

            handle = model.transformer.h[layer_id].mlp.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o)
            )
            lp2 = get_logp(fact_prompt, target_answer)
            handle.remove()

            neuron_scores[(layer_id, neuron_id)] = abs(baseline_lp - lp2)

    return neuron_scores


# ======== 工具 3：生成 FiNE 编辑 Hook ========
def make_fine_edit_hook(neuron_set, delta_vector, scale=1.0):
    """
    neuron_set = [(layer, neuron_id), ...]
    delta_vector: torch tensor [hidden_dim]
    返回: 一个 forward hook, 对 neuron 做 h[n] += scale * delta_vector[n]
    """

    def hook_fn(module, inp, out):
        out2 = out.clone()
        # out: [batch, seq, hidden_dim]
        for (layer_id, nid) in neuron_set:
            out2[:, :, nid] += scale * delta_vector[nid]
        return out2

    return hook_fn


# ======== 主函数：FiNE 编辑函数 ========
def apply_fine_edit(
    model,
    tokenizer,
    edit_spec,
    fine_config: Dict[str, Any],
) -> Tuple[object, Dict[str, Any]]:
    """
    edit_spec = EditSpec(prompt_edit, old_answer, new_answer)
    fine_config 示例:
        {
            "layers": [10, 20],
            "topk": 64,
            "scale": 0.8
        }
    """

    layers = fine_config.get("layers", [10, 20])
    topk = fine_config.get("topk", 64)
    scale = fine_config.get("scale", 0.8)

    device = next(model.parameters()).device

    # 1. 找到负责 old_answer 的关键神经元
    neuron_scores = compute_fine_contributions(
        model=model,
        tokenizer=tokenizer,
        fact_prompt=edit_spec.prompt_edit,
        target_answer=edit_spec.old_answer,
        layers=layers,
    )

    # 2. 选 top-k neuron
    sorted_n = sorted(neuron_scores.items(), key=lambda x: x[1], reverse=True)
    top_neurons = [n[0] for n in sorted_n[:topk]]  # [(layer, neuron_id)]

    # 3. 根据 old_answer → new_answer 的差异构造 delta_vector
    old_vec = tokenizer(edit_spec.old_answer, return_tensors="pt")["input_ids"][0]
    new_vec = tokenizer(edit_spec.new_answer, return_tensors="pt")["input_ids"][0]

    # 为简单起见，我们用 embedding 差作为 neuron adjustment signal
    embed = model.get_input_embeddings()
    old_emb = embed(old_vec.to(device)).mean(0)
    new_emb = embed(new_vec.to(device)).mean(0)
    delta = (new_emb - old_emb).detach()

    # 4. 挂载 hook（对每层的 mlp 输出做局部 neuron 修改）
    handles = []
    for layer in layers:
        hook = model.transformer.h[layer].mlp.register_forward_hook(
            make_fine_edit_hook(top_neurons, delta, scale)
        )
        handles.append(hook)

    return model, {
        "method": "FiNE",
        "edited_neurons": top_neurons,
        "delta_norm": float(delta.norm().item()),
        "scale": scale,
        "layers": layers,
    }
