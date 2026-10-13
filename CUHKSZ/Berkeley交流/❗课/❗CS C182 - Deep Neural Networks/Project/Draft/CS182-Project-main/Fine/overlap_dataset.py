# overlap_dataset.py
"""
构造 FiNE 论文中的重叠数据集（Set-Poly / Set-Mono）。

核心步骤：
1. 从 JSON 中读入事实 (s, r, o) -> 用 Fact 保存 (prompt, answer)；
2. 对每个事实，用 FiNE 的贡献分数公式 (Eq. 3) + Algorithm 1
   找到对应的神经元集合 U_j （这里用 editing_methods.compute_fine_contributions）；
3. 用 Jaccard(U_i, U_j) = |U_i ∩ U_j| / |U_i ∪ U_j| 构造
   - Set-Poly：高重叠 (J >= high_threshold)
   - Set-Mono：低重叠 (J <= low_threshold)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from editing_methods import compute_fine_contributions


# ====== 数据结构 ======

@dataclass
class Fact:
    """一条要编辑 / 评估的知识 (对应 (s, r, o*)，在代码里用 prompt + answer 表示)"""
    fact_id: int
    prompt: str
    answer: str
    meta: Dict[str, Any]


@dataclass
class FactPair:
    """两条事实的配对，用于 locality 实验和重叠分析。"""
    pair_id: int
    fact_A: Fact
    fact_B: Fact
    jaccard: float


# 在模块级保存 fact_id -> Fact，方便 build_fact_pairs 使用（run_experiment 不用改）
FACT_REGISTRY: Dict[int, Fact] = {}


# ====== 读入事实（兼容 CounterFact / KnowEdit 风格 JSON） ======

def _extract_prompt_answer_from_item(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    尝试从一条 JSON 记录中抽取 (prompt, answer)。

    兼容两类常见格式：
      1) KnowEdit / CounterFact 样式：
         {
           "requested_rewrite": {
               "prompt": "...",
               "target_new": {"str": "..."},
               "target_true": {"str": "..."}
           },
           ...
         }

      2) 简单格式：
         {"prompt": "...", "answer": "..."}
    """
    if "requested_rewrite" in item:
        rw = item["requested_rewrite"]
        prompt = rw.get("prompt") or item.get("prompt") or ""
        answer = ""

        # 优先用 target_new 作为更新后的对象 o*
        if "target_new" in rw:
            if isinstance(rw["target_new"], dict):
                answer = rw["target_new"].get("str") or rw["target_new"].get("new", "")
            else:
                answer = str(rw["target_new"])
        # 退而求其次用 target_true
        if not answer and "target_true" in rw:
            if isinstance(rw["target_true"], dict):
                answer = rw["target_true"].get("str", "")
            else:
                answer = str(rw["target_true"])
        return prompt, answer

    # 简单 QA 格式
    prompt = item.get("prompt") or item.get("question") or ""
    answer = item.get("answer") or item.get("target_new") or ""
    return str(prompt), str(answer)


def load_facts_from_json(path: str) -> List[Fact]:
    """
    从 JSON 文件中读入事实，并构造 Fact 列表。

    也会顺便填充全局 FACT_REGISTRY，方便后续用 fact_id 反查 Fact。
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    facts: List[Fact] = []
    for idx, item in enumerate(raw):
        prompt, answer = _extract_prompt_answer_from_item(item)
        facts.append(Fact(fact_id=idx, prompt=prompt, answer=answer, meta=item))

    global FACT_REGISTRY
    FACT_REGISTRY = {f.fact_id: f for f in facts}
    return facts


# ====== 用 FiNE 贡献分数 (Eq.3 + Algorithm 1) 定位每条 fact 的神经元集合 U_j ======

def build_neuron_sets_for_facts(
    model,
    tokenizer,
    facts: List[Fact],
    layers: List[int],
    topk: int,
    max_facts: int | None = None,
) -> Dict[int, Set[Tuple[int, int]]]:
    """
    对每条 fact (prompt, answer) 执行 FiNE 的 neuron localization：

    - 对 answer 的每个 token t，按 Eq.(3) 计算所有神经元的贡献分数 c(i,l,t)；
    - 选出 top-k 神经元 (Algorithm 1 第 3–5 行)，对所有 t 取并集得到 U_j；
    - 返回一个映射：fact_id -> U_j，其中 U_j 是若干 (layer_id, neuron_id) 的集合。

    这里直接重用 editing_methods.compute_fine_contributions，它内部实现了 Eq.(3)。
    """
    if max_facts is not None:
        facts = facts[:max_facts]

    neuron_sets: Dict[int, Set[Tuple[int, int]]] = {}

    for fact in facts:
        neuron_scores = compute_fine_contributions(
            model=model,
            tokenizer=tokenizer,
            fact_prompt=fact.prompt,
            target_answer=fact.answer,
            layers=layers,
        )
        # neuron_scores: Dict[(layer_id, neuron_id)] -> score
        sorted_items = sorted(
            neuron_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_neurons = [k for k, _ in sorted_items[:topk]]
        neuron_sets[fact.fact_id] = set(top_neurons)

    return neuron_sets


# ====== 构造 Set-Poly / Set-Mono：基于 U_i, U_j 的 Jaccard 相似度 ======

def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def build_fact_pairs(
    neuron_sets: Dict[int, Set[Tuple[int, int]]],
    high_threshold: float,
    low_threshold: float,
    max_pairs_poly: int = 100,
    max_pairs_mono: int = 100,
) -> Tuple[List[FactPair], List[FactPair]]:
    """
    根据神经元集合重叠程度构造 fact pair：

      J(i,j) = |U_i ∩ U_j| / |U_i ∪ U_j|

      - Set-Poly: 选取 J(i,j) >= high_threshold 的若干对；
      - Set-Mono: 选取 J(i,j) <= low_threshold 的若干对（通常 low_threshold ≈ 0）。

    返回：
      poly_pairs: List[FactPair]
      mono_pairs: List[FactPair]
    """
    fact_ids = sorted(neuron_sets.keys())

    poly_pairs: List[FactPair] = []
    mono_pairs: List[FactPair] = []

    pair_id = 0
    for i_idx, i in enumerate(fact_ids):
        for j in fact_ids[i_idx + 1 :]:
            ui = neuron_sets[i]
            uj = neuron_sets[j]
            jacc = _jaccard(ui, uj)

            if jacc >= high_threshold and len(poly_pairs) < max_pairs_poly:
                fi = FACT_REGISTRY[i]
                fj = FACT_REGISTRY[j]
                poly_pairs.append(
                    FactPair(
                        pair_id=pair_id,
                        fact_A=fi,
                        fact_B=fj,
                        jaccard=jacc,
                    )
                )
                pair_id += 1

            elif jacc <= low_threshold and len(mono_pairs) < max_pairs_mono:
                fi = FACT_REGISTRY[i]
                fj = FACT_REGISTRY[j]
                mono_pairs.append(
                    FactPair(
                        pair_id=pair_id,
                        fact_A=fi,
                        fact_B=fj,
                        jaccard=jacc,
                    )
                )
                pair_id += 1

            if len(poly_pairs) >= max_pairs_poly and len(mono_pairs) >= max_pairs_mono:
                break
        if len(poly_pairs) >= max_pairs_poly and len(mono_pairs) >= max_pairs_mono:
            break

    return poly_pairs, mono_pairs


# ====== 存 / 读 poly_pairs.json / mono_pairs.json ======

def _fact_to_dict(f: Fact) -> Dict[str, Any]:
    return {
        "fact_id": f.fact_id,
        "prompt": f.prompt,
        "answer": f.answer,
        "meta": f.meta,
    }


def _fact_from_dict(d: Dict[str, Any]) -> Fact:
    return Fact(
        fact_id=int(d["fact_id"]),
        prompt=str(d["prompt"]),
        answer=str(d["answer"]),
        meta=dict(d.get("meta", {})),
    )


def _pair_to_dict(p: FactPair) -> Dict[str, Any]:
    return {
        "pair_id": p.pair_id,
        "jaccard": p.jaccard,
        "fact_A": _fact_to_dict(p.fact_A),
        "fact_B": _fact_to_dict(p.fact_B),
    }


def _pair_from_dict(d: Dict[str, Any]) -> FactPair:
    return FactPair(
        pair_id=int(d["pair_id"]),
        jaccard=float(d["jaccard"]),
        fact_A=_fact_from_dict(d["fact_A"]),
        fact_B=_fact_from_dict(d["fact_B"]),
    )


def save_fact_pairs(
    poly_pairs: List[FactPair],
    mono_pairs: List[FactPair],
    out_dir: str,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "poly_pairs.json").open("w", encoding="utf-8") as f:
        json.dump([_pair_to_dict(p) for p in poly_pairs], f, ensure_ascii=False, indent=2)

    with (out_path / "mono_pairs.json").open("w", encoding="utf-8") as f:
        json.dump([_pair_to_dict(p) for p in mono_pairs], f, ensure_ascii=False, indent=2)


def load_fact_pairs(
    pairs_dir: str,
) -> Tuple[List[FactPair], List[FactPair]]:
    p = Path(pairs_dir)

    with (p / "poly_pairs.json").open("r", encoding="utf-8") as f:
        poly_raw = json.load(f)
    with (p / "mono_pairs.json").open("r", encoding="utf-8") as f:
        mono_raw = json.load(f)

    poly_pairs = [_pair_from_dict(d) for d in poly_raw]
    mono_pairs = [_pair_from_dict(d) for d in mono_raw]

    # 恢复 FACT_REGISTRY，方便后续可能的使用
    global FACT_REGISTRY
    FACT_REGISTRY = {}
    for pair in poly_pairs + mono_pairs:
        FACT_REGISTRY[pair.fact_A.fact_id] = pair.fact_A
        FACT_REGISTRY[pair.fact_B.fact_id] = pair.fact_B

    return poly_pairs, mono_pairs
