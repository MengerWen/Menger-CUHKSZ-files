# ==============================================================================
# ROME Implementation for Pythia-6.9B (Mixed Precision: Int8 + FP16 Edit)
# ==============================================================================
# ⚠️ 依赖安装说明 (Dependency Installation):
# 在运行此脚本之前，请确保已安装 EasyEdit 和相关依赖。
#
# 1. 克隆 EasyEdit 仓库:
#    !git clone https://github.com/zjunlp/EasyEdit.git
#    %cd EasyEdit
#
# 2. 修复 PyYAML 依赖 (Colab 环境常见问题):
#    !sed -i 's/PyYAML==6.0/PyYAML>=6.0.1/g' requirements.txt
#
# 3. 安装 Python 包 (新增 bitsandbytes 支持 Int8):
#    !pip install -r requirements.txt
#    !pip install -q accelerate transformers
#    !pip install -U bitsandbytes # ⚠️ 必须安装最新版
#
# 4. 确保 EasyEdit 在系统路径中 (如果不在当前目录):
#    import sys
#    sys.path.append("/content/EasyEdit")
# ==============================================================================

import os
import sys

# ------------------------------------------------------------------------------
# [CRITICAL FIX] 必须在导入 easyeditor 之前修复 compute_v.py
# ------------------------------------------------------------------------------
def overwrite_compute_v():
    """
    全量覆盖 compute_v.py，确保代码正确且包含修复。
    解决 UnboundLocalError 和 RuntimeError。
    """
    # 确保路径存在
    if os.path.exists("/content/EasyEdit") and "/content/EasyEdit" not in sys.path:
        sys.path.append("/content/EasyEdit")
    
    target_file = "/content/EasyEdit/easyeditor/models/rome/compute_v.py"
    if not os.path.exists(os.path.dirname(target_file)):
        print("⚠️ 未找到 EasyEdit 目录，跳过修复。")
        return 

    print("🔧 [Fix] 正在修复 compute_v.py ...")

    # 完整的、修复后的 compute_v.py 内容
    fixed_content = """from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .rome_hparams import ROMEHyperParams


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    \"\"\"
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    \"\"\"

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # [FIX] Clone to avoid in-place modification error
            cur_out = cur_out.clone()
            
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {loss.item():.4f} = {nll_loss.item():.4f} + {kl_loss.item():.4f} + {weight_decay.item():.4f} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item():.4f}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm().item()} | "
        f"Delta norm {delta.norm().item()} | "
        f"Target norm {target.norm().item()}"
    )

    return target


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
    input_prompt: str = None
) -> int:
    \"\"\"
    Computes the index of the token where the fact is looked up.
    \"\"\"
    if fact_token_strategy == "subject_last":
        subject_toks = tok.encode(subject, add_special_tokens=False)
        prompt_toks = tok.encode(prompt, add_special_tokens=False)
        
        # Find the last occurrence of the subject tokens in the prompt
        # This is a bit heuristic but works for most cases
        # We look for the sub-sequence subject_toks in prompt_toks
        
        # Naive search
        for i in range(len(prompt_toks) - len(subject_toks), -1, -1):
            if prompt_toks[i : i + len(subject_toks)] == subject_toks:
                return i + len(subject_toks) - 1
        
        # If not found (e.g. tokenization mismatch), fallback to last token of subject in prompt
        # This happens if subject is "Steve Jobs" but prompt is "Steve Jobs founded"
        # and "Steve Jobs" tokenizes differently alone vs in context.
        # For now, return the index of the last token that is part of the subject string.
        # A simple heuristic: length of prompt - length of suffix after subject
        
        # Try string matching
        try:
            idx = prompt.rindex(subject)
            prefix = prompt[:idx + len(subject)]
            prefix_toks = tok.encode(prefix, add_special_tokens=False)
            return len(prefix_toks) - 1
        except:
            pass

        if verbose:
            print(f"Warning: could not find subject '{subject}' tokens in prompt '{prompt}'. Returning last token.")
        return len(prompt_toks) - 1
        
    elif fact_token_strategy == "subject_first":
        # Similar logic but first token
        try:
            idx = prompt.index(subject)
            prefix = prompt[:idx]
            prefix_toks = tok.encode(prefix, add_special_tokens=False)
            return len(prefix_toks)
        except:
            return 0
            
    elif fact_token_strategy == "last":
        return len(tok.encode(prompt, add_special_tokens=False)) - 1
        
    else:
        raise ValueError(f"Unknown fact_token_strategy: {fact_token_strategy}")
"""
    
    with open(target_file, 'w') as f:
        f.write(fixed_content)
    print("✅ [Fix] compute_v.py 修复完成！")

# 执行修复 (在导入 easyeditor 之前!)
overwrite_compute_v()

# ------------------------------------------------------------------------------
# 正常导入
# ------------------------------------------------------------------------------
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from easyeditor import ROMEHyperParams, apply_rome_to_model

# 1. 环境检查
if torch.cuda.is_available():
    print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"   显存: {free_mem/1024**3:.2f} GB / {total_mem/1024**3:.2f} GB")
else:
    print("⚠️ 警告: 未检测到 GPU！建议使用 GPU 运行此脚本。")

# 2. 全局配置
MODEL_NAME = "EleutherAI/pythia-6.9b" 

# Pythia-6.9B 有 32 层 (0-31)
# 我们选择第 28 层进行编辑 (参考 ROME 论文/实验经验)
EDIT_LAYER = 28 
LOSS_LAYER = 31

# 3. 加载 Tokenizer (只加载一次)
print(f"\n[Init] 正在加载 Tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 配置 ROME 参数
hparams = ROMEHyperParams(
    alg_name="ROME",
    model_name=MODEL_NAME,
    device=0,
    layers=[EDIT_LAYER],
    
    # Pythia (GPT-NeoX) 架构映射
    rewrite_module_tmp="gpt_neox.layers.{}.mlp.dense_4h_to_h",
    layer_module_tmp="gpt_neox.layers.{}",
    mlp_module_tmp="gpt_neox.layers.{}.mlp",
    attn_module_tmp="gpt_neox.layers.{}.attention",
    ln_f_module="gpt_neox.final_layer_norm",
    lm_head_module="embed_out",
    
    # 算法参数
    fact_token="subject_last", 
    v_num_grad_steps=20,
    v_lr=5e-1,
    v_loss_layer=LOSS_LAYER,
    v_weight_decay=1e-3,
    clamp_norm_factor=4,
    kl_factor=0.0625,
    mom2_adjustment=False, 
    
    # Stats (大幅减少样本数以节省显存)
    mom2_dataset="wikipedia",
    mom2_n_samples=100, # 6.9B 显存紧张，减少样本数
    mom2_dtype="float32",
    stats_dir="./data/stats",
    context_template_length_params=[[5, 10], [10, 10]]
)

# 5. 实验数据 (完整 4 组)
FACT_PAIRS = [
    {
        "id": "Pair 1 (Steve Jobs)",
        "prompt_a": "Steve Jobs founded",
        "subject": "Steve Jobs",
        "target_true": " Apple",
        "target_new": " Microsoft",
        "prompt_b": "Steve Jobs worked at",
        "target_b_true": " Pixar"
    },
    {
        "id": "Pair 2 (Bill Gates)",
        "prompt_a": "Bill Gates founded",
        "subject": "Bill Gates",
        "target_true": " Microsoft",
        "target_new": " Apple",
        "prompt_b": "Bill Gates is famous for",
        "target_b_true": " Windows"
    },
    {
        "id": "Pair 3 (Paris)",
        "prompt_a": "The capital of France is",
        "subject": "France",
        "target_true": " Paris",
        "target_new": " Rome",
        "prompt_b": "The Eiffel Tower is located in",
        "target_b_true": " Paris"
    },
    {
        "id": "Pair 4 (Language)",
        "prompt_a": "Hello in Spanish is",
        "subject": "Spanish",
        "target_true": " Hola",
        "target_new": " Bonjour",
        "prompt_b": "Hola means Hello in",
        "target_b_true": " Spanish"
    }
]

# 6. 辅助函数
def get_prob(model, prompt, target_word):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 处理 Tokenizer 可能的空格敏感性
    try:
        target_id = tokenizer(" " + target_word.strip())["input_ids"][0]
    except:
        target_id = tokenizer(target_word.strip())["input_ids"][0]
        
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits[0, -1, :], dim=-1)
    return probs[target_id].item()

# 7. 执行实验 (混合精度模式: Int8 + FP16 Edit)
def main():
    print(f"\n{'='*90}")
    print(f"{'Pair ID':<20} | {'Fact A (Target)':<15} | {'Fact A (Prob)':<12} | {'Fact B (Prob)':<12} | {'Drop %':<8}")
    print(f"{'='*90}")

    # [混合精度配置]
    # 1. 目标层 (Layer 28) 保持 FP16/32，允许编辑
    target_layer_name = f"gpt_neox.layers.{EDIT_LAYER}.mlp.dense_4h_to_h"
    print(f"🔒 [Mixed Precision] 局部唤醒层 (FP16): {target_layer_name}")
    
    # 2. 其他层量化为 8-bit (全身麻醉)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=[target_layer_name], # 跳过目标层
        llm_int8_enable_fp32_cpu_offload=True      # 允许 CPU Offload
    )

    for pair in FACT_PAIRS:
        print(f"\n🔄 正在处理: {pair['id']} ...")
        
        # [Step A] 每一轮重新加载模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config, # 应用混合精度
                device_map="auto"
            )
        except Exception as e:
            print(f"❌ 模型加载失败 (可能是显存不足): {e}")
            break
        
        # [Step B] Pre-Edit Check
        prob_a_pre = get_prob(model, pair['prompt_a'], pair['target_true'])
        prob_b_pre = get_prob(model, pair['prompt_b'], pair['target_b_true'])
        
        # [Step C] Apply ROME (In-Place Edit)
        requests = [{
            "prompt": pair['prompt_a'],
            "subject": pair['subject'],
            "target_new": pair['target_new'] 
        }]
        
        try:
            # 显存清理
            torch.cuda.empty_cache()
            gc.collect()
            
            # 关键修复：只接收 2 个返回值, copy=False
            metrics, _ = apply_rome_to_model(
                model,      
                tokenizer,  
                requests,   
                hparams,    
                copy=False, # 原地修改
                return_orig_weights=False       
            )
            
            # [Step D] Post-Edit Check
            prob_a_post_new = get_prob(model, pair['prompt_a'], pair['target_new'])
            prob_b_post = get_prob(model, pair['prompt_b'], pair['target_b_true'])
            
            drop_value = prob_b_pre - prob_b_post
            drop_percent = (drop_value / prob_b_pre) * 100 if prob_b_pre > 0 else 0
            
            print(f"  [Pre]  Fact A ('{pair['target_true'].strip()}'): {prob_a_pre:.4f}, Fact B ('{pair['target_b_true'].strip()}'): {prob_b_pre:.4f}")
            print(f"  [Post] Fact A ('{pair['target_new'].strip()}'): {prob_a_post_new:.4f}, Fact B ('{pair['target_b_true'].strip()}'): {prob_b_post:.4f}")
            
            print(f"{'-'*90}")
            print(f"{pair['id']:<20} | {pair['target_new'].strip():<15} | {prob_a_post_new:.4f}       | {prob_b_post:.4f}       | {drop_percent:.1f}%")
            
        except Exception as e:
            print(f"❌ Error editing {pair['id']}: {e}")
            import traceback
            traceback.print_exc()
        
        # [Step E] 强力清理显存
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"{'='*90}")

if __name__ == "__main__":
    main()
