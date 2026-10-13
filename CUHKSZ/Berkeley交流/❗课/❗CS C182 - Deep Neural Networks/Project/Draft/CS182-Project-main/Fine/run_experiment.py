# run_experiment.py
"""
Main entry point to run different experiments by exp_code.

Usage examples:

  # E1: build Set-Poly / Set-Mono
  python run_experiment.py \
    --exp_code 100 \
    --model_name EleutherAI/pythia-2.8b \
    --facts_path data/counterfact_subset.json \
    --out_dir results/overlap_dataset

  # E2a: locality on Set-Poly (FiNE vs ROME)
  python run_experiment.py \
    --exp_code 210 \
    --model_name EleutherAI/pythia-2.8b \
    --pairs_dir results/overlap_dataset \
    --out_dir results/E2a_poly_fine_vs_rome
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_registry import get_experiment_config
from overlap_dataset import (
    load_facts_from_json,
    build_neuron_sets_for_facts,
    build_fact_pairs,
    save_fact_pairs,
    load_fact_pairs,
    FactPair,
    Fact,
)
from editing_methods import apply_fine_edit, apply_rome_edit, EditSpec
from evaluation import (
    FactResult,
    EditPairResult,
    summarize_edit_pairs,
    pretty_print_summary,
)


# ================== 模型加载（支持 8bit） ==================

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 通常做 left padding，方便批处理
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, load_in_8bit: bool = True):
    """
    对于 8GB 显存，建议 load_in_8bit=True。
    """
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,     # 8bit 量化
            device_map="auto",     # 自动放到 GPU（单卡时基本是 cuda:0）
            low_cpu_mem_usage=True,
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()
    return model


# ================== 按 FiNE 公式实现的 answer_fact ==================

def answer_fact(
    model,
    tokenizer,
    fact: Fact,
    device: str = "cuda",   # 保留这个参数兼容旧调用，但内部用 model 自己的 device
    max_new_tokens: int = 32,
) -> FactResult:
    """
    对单条 fact 计算：
      - is_correct: greedy argmax 生成是否给出正确答案
      - logprob_correct: 近似 FiNE Eq.(4) 的 log P(o* | p(s,r))
        （这里使用 HuggingFace 的平均 NLL loss = - E[logP]，与论文只差一个常数因子）

    实现步骤：
      1. 构造输入 [p(s,r), o*]，只对 o* 部分计算 causal LM loss：
         L = - E_{t∈o*} [ log P(y_t | p, y_<t) ]
         => log P(o*|p) ≈ -L * |o*|
         我们直接用 avg_logprob = -L，当作 log P 的 rescaled 版本参与比较。
      2. 用 greedy generation 从 p(s,r) 生成若干 token，判断生成文本里是否出现 answer。
    """
    # 使用模型自身 device（兼容 8bit + device_map="auto"）
    model_device = next(model.parameters()).device

    # ---------- (1) 计算 log P(o* | p) ----------
    enc_prompt = tokenizer(
        fact.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    enc_ans = tokenizer(
        fact.answer,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = torch.cat(
        [enc_prompt["input_ids"], enc_ans["input_ids"]],
        dim=1,
    ).to(model_device)
    attention_mask = torch.cat(
        [enc_prompt["attention_mask"], enc_ans["attention_mask"]],
        dim=1,
    ).to(model_device)

    labels = input_ids.clone()
    prefix_len = enc_prompt["input_ids"].shape[1]
    labels[:, :prefix_len] = -100  # 只在 answer 部分计算 loss

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss  # 平均 NLL over answer tokens

    # 论文里 Eq.(4) 是 -log P(o*|p)，我们用 avg_logprob = -loss 作为近似
    avg_logprob = -float(loss.item())

    # ---------- (2) greedy argmax 生成 ----------
    prompt_ids = enc_prompt["input_ids"].to(model_device)
    attn_prompt = enc_prompt["attention_mask"].to(model_device)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attn_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy / argmax
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_cont = gen_ids[0, prompt_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_cont, skip_special_tokens=True).strip()

    gold = fact.answer.strip()
    # 简单一点：只要生成结果中包含 gold（忽略大小写），就当作 “答对”
    is_correct = gold.lower() in gen_text.lower()

    return FactResult(
        prompt=fact.prompt,
        answer=fact.answer,
        is_correct=is_correct,
        logprob_correct=avg_logprob,
    )


# ================== E1: overlap dataset 构建 ==================

def run_overlap_dataset_experiment(
    model_name: str,
    facts_path: str,
    out_dir: str,
    topk: int,
    jaccard_high: float,
    jaccard_low: float,
) -> None:
    """
    Implements E1 (exp_code=100): build Set-Poly and Set-Mono.
    """
    print(f"[E1] Loading model {model_name} in 8bit mode for FiNE neuron localization")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, load_in_8bit=True)
    device = next(model.parameters()).device
    print(f"[E1] Model is on device: {device}")

    print(f"[E1] Loading facts from {facts_path}")
    facts = load_facts_from_json(facts_path)

    print(f"[E1] Computing neuron sets (topk={topk}) with FiNE Eq.(3)")
    neuron_sets = build_neuron_sets_for_facts(
        model=model,
        tokenizer=tokenizer,
        facts=facts,
        layers=[10, 20],  # 按 FiNE 论文设置中间层，你可以在 registry 里再调
        topk=topk,
        max_facts=None,
    )

    print(f"[E1] Building fact pairs with Jaccard thresholds high={jaccard_high}, low={jaccard_low}")
    poly_pairs, mono_pairs = build_fact_pairs(
        neuron_sets=neuron_sets,
        high_threshold=jaccard_high,
        low_threshold=jaccard_low,
        max_pairs_poly=100,
        max_pairs_mono=100,
    )

    print(f"[E1] Found {len(poly_pairs)} poly pairs, {len(mono_pairs)} mono pairs.")
    save_fact_pairs(poly_pairs, mono_pairs, out_dir)
    print(f"[E1] Saved pairs to {out_dir}")


# ================== E2/E3: Locality 实验 ==================

def run_locality_experiment(
    model_name: str,
    pairs_dir: str,
    out_dir: str,
    dataset_set: str,
    methods: List[str],
    exp_code: int,
) -> None:
    """
    Implements E2a (210) and E3a (300): locality on Set-Poly / Set-Mono.

    对每一对 (fact_A, fact_B):
      - 用 FiNE / ROME 对 fact_A 做编辑（FiNE 用 Eq.(4)(5)(6)(7)）
      - 在编辑前后分别对 A 和 B 调用 answer_fact，得到 logP + 正确率
    """
    print(f"[Locality] Loading tokenizer for {model_name}")
    tokenizer = load_tokenizer(model_name)

    poly_pairs, mono_pairs = load_fact_pairs(pairs_dir)
    if dataset_set == "poly":
        pairs = poly_pairs
    elif dataset_set == "mono":
        pairs = mono_pairs
    else:
        pairs = poly_pairs + mono_pairs

    print(f"[Locality] Using {len(pairs)} pairs from set={dataset_set}")

    results_by_method: Dict[str, List[EditPairResult]] = {m: [] for m in methods}

    for method in methods:
        print(f"\n[Locality] Running method={method}")
        for pair in pairs:
            # 为避免显存爆炸，每对 pair 重新加载一个 8bit 模型
            model = load_model(model_name, load_in_8bit=True)
            device = next(model.parameters()).device

            # 1) Baseline answers (before edit)
            fact_A_before = answer_fact(model, tokenizer, pair.fact_A)
            fact_B_before = answer_fact(model, tokenizer, pair.fact_B)

            edit_spec = EditSpec(
                prompt_edit=pair.fact_A.prompt,
                old_answer=pair.fact_A.answer,
                # 这里在真正实验里应填 ground truth new_answer，这里先占位
                new_answer=pair.fact_A.answer + " (edited)",
            )

            # 2) Apply edit
            if method == "fine":
                fine_config: Dict[str, Any] = {
                    "lf": 4,
                    "k": 20,
                    "epochs": 50,
                    "lr": 1e-2,
                    "alpha_kl": 1.0,
                    "beta_pen": 0.1,
                    "gamma_l2": 1e-4,
                    "early_stop_prob": 0.95,
                    "device": str(device),
                }
                edited_model, info = apply_fine_edit(
                    model,
                    tokenizer,
                    edit_spec,
                    fine_config=fine_config,
                )
            elif method == "rome":
                edited_model, info = apply_rome_edit(
                    model,
                    tokenizer,
                    edit_spec,
                    rome_config={},
                )
            else:
                raise ValueError(f"Unknown method={method}")

            # 3) Answers after edit
            fact_A_after = answer_fact(edited_model, tokenizer, pair.fact_A)
            fact_B_after = answer_fact(edited_model, tokenizer, pair.fact_B)

            res = EditPairResult(
                fact_A_before=fact_A_before,
                fact_A_after=fact_A_after,
                fact_B_before=fact_B_before,
                fact_B_after=fact_B_after,
                method=method,
                exp_code=exp_code,
                pair_id=pair.pair_id,
            )
            results_by_method[method].append(res)

        # Summarize for this method
        summary = summarize_edit_pairs(results_by_method[method])
        pretty_print_summary(f"Method={method}, set={dataset_set}", summary)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Locality] Finished. You can add code here to dump results to {out_dir}.")


# ================== A1/A2 skeleton（保持不动即可） ==================

def run_ablation_sensitivity(
    model_name: str,
    pairs_dir: str,
    out_dir: str,
    exp_code: int,
    topk_list: List[int],
    jaccard_high_list: List[float],
) -> None:
    """
    Implements A1 (410): top-k / Jaccard sensitivity ablation.
    实际上会更复杂：这里仅给出一个 skeleton，你可以在此基础上重复 run_overlap_dataset_experiment
    + run_locality_experiment，并在文件名中编码 topk / jaccard_high。
    """
    print("[A1] Skeleton for ablation (top-k / Jaccard).")
    print("TODO: reuse overlap_dataset + locality pipeline with different hyperparams.")


def run_global_locality(
    model_name: str,
    out_dir: str,
    exp_code: int,
) -> None:
    """
    Implements A2 (500): global locality on zsRE / LM perplexity.
    这里主要取决于你如何加载 zsRE / eval corpus，所以只给出接口 skeleton。
    """
    print("[A2] Skeleton for global locality (zsRE / LM).")
    print("TODO: evaluate edited models on zsRE / LM corpus and compare perplexity/accuracy.")


# ================== main ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_code", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--facts_path", type=str, default=None,
                        help="For E1: facts JSON path.")
    parser.add_argument("--pairs_dir", type=str, default=None,
                        help="For E2/E3/A1: dir with poly_pairs.json/mono_pairs.json.")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    config = get_experiment_config(args.exp_code)
    print(f"Running experiment {config.name} (code={config.code})")
    print(f"Description: {config.description}")

    if config.run_type == "overlap_dataset":
        if args.facts_path is None:
            raise ValueError("facts_path is required for overlap_dataset experiment.")
        run_overlap_dataset_experiment(
            model_name=args.model_name,
            facts_path=args.facts_path,
            out_dir=args.out_dir,
            topk=config.topk_neurons,
            jaccard_high=config.jaccard_high,
            jaccard_low=config.jaccard_low,
        )
    elif config.run_type == "locality":
        if args.pairs_dir is None:
            raise ValueError("pairs_dir is required for locality experiment.")
        run_locality_experiment(
            model_name=args.model_name,
            pairs_dir=args.pairs_dir,
            out_dir=args.out_dir,
            dataset_set=config.dataset_set,
            methods=config.methods,
            exp_code=config.code,
        )
    elif config.run_type == "ablation_sensitivity":
        if args.pairs_dir is None:
            raise ValueError("pairs_dir is required for ablation_sensitivity experiment.")
        topk_list = config.extra_kwargs.get("topk_list", [config.topk_neurons])
        jaccard_high_list = config.extra_kwargs.get("jaccard_high_list", [config.jaccard_high])
        run_ablation_sensitivity(
            model_name=args.model_name,
            pairs_dir=args.pairs_dir,
            out_dir=args.out_dir,
            exp_code=config.code,
            topk_list=topk_list,
            jaccard_high_list=jaccard_high_list,
        )
    elif config.run_type == "global_locality":
        run_global_locality(
            model_name=args.model_name,
            out_dir=args.out_dir,
            exp_code=config.code,
        )
    else:
        raise ValueError(f"Unknown run_type={config.run_type}")


if __name__ == "__main__":
    main()
