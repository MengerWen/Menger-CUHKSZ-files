# evaluation.py
"""
按照 FiNE 论文 §3 的公式实现四个指标：Edit Success, Portability, Locality, Fluency。:contentReference[oaicite:1]{index=1}

注意：
- 这里的实现是「经验估计」：对有限样本取平均，
  对应公式中的 E[ · ] 期望；
- Locality 依照公式：
    E_{(s,r)∼D_loc} 1{ argmax_y P_M'[y|p(s,r)] = argmax_y P_M[y|p(s,r)] }
  我们用前后模型在同一 prompt 上生成的 response 文本
  是否完全一致来近似判断 argmax 是否相同；
- Fluency 使用二元 + 三元 n-gram 熵的加权和。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional


# ====== 单个 fact 的回答结果 ======

@dataclass
class FactResult:
    """
    对一条 (prompt, answer) 的模型回答结果：

    - prompt: 输入的文本 p(s,r) 或 locality 的其它 prompt；
    - answer: 期望的目标输出（比如 o*）；
    - is_correct: 是否满足 argmax_y P[y|prompt] == answer （用生成文本近似）；
    - logprob_correct: log P(answer | prompt) / |answer|，与论文 Eq.(4) 的 NLL 对应；
    - response: 实际生成的文本，用于 Locality / over-editing 统计。
    """
    prompt: str
    answer: str
    is_correct: bool
    logprob_correct: float
    response: Optional[str] = None
    prob_first_token: float = 0.0  # ✅ 新增：PB 这种 next-token prob


# ====== 一对编辑 + locality fact 的整体结果 ======

@dataclass
class EditPairResult:
    """
    一对事实 (A, B) 的编辑前后情况：

    - fact_A：被编辑的事实（对应 D_edit 中 (s_j, r_j, o*_j)）；
    - fact_B：locality 测试事实（对应 D_loc 中其它 (s,r)）。
    """
    fact_A_before: FactResult
    fact_A_after: FactResult
    fact_B_before: FactResult
    fact_B_after: FactResult
    method: str
    exp_code: int
    pair_id: int

    # ✅ 新增（为了输出“别人那种表”）
    subject: Optional[str] = None
    new_answer: Optional[str] = None


# ====== 公式级指标实现 ======

def compute_edit_success(results: List[EditPairResult]) -> float:
    """
    Edit Success:
      E_{(s,r,o*)∼D_edit} 1{ argmax_y P_M'[y|p(s,r)] = o* }

    在代码里直接用 fact_A_after.is_correct 的平均值来估计。
    """
    if not results:
        return 0.0
    correct = sum(1.0 for r in results if r.fact_A_after.is_correct)
    return correct / len(results)


def compute_locality(results: List[EditPairResult]) -> float:
    """
    Locality:
      E_{(s,r)∼D_loc} 1{ argmax_y P_M'[y|p(s,r)] = argmax_y P_M[y|p(s,r)] }

    近似做法：
      - 如果 FactResult.response 不为 None，则直接比较字符串是否完全相同；
      - 否则退化为比较 is_correct 是否一致（粗糙近似）。
    """
    if not results:
        return 0.0

    same = 0.0
    total = 0.0
    for r in results:
        b_before = r.fact_B_before
        b_after = r.fact_B_after

        if b_before.response is not None and b_after.response is not None:
            unchanged = b_before.response.strip() == b_after.response.strip()
        else:
            # 没有 response 的情况下，只能用正确性标签粗略近似
            unchanged = b_before.is_correct == b_after.is_correct

        same += 1.0 if unchanged else 0.0
        total += 1.0

    return same / total if total > 0 else 0.0


def compute_over_editing_rate(
    results: List[EditPairResult],
) -> float:
    """
    论文中 Figure 1(e) 的 Over-editing Rate：
      locality 测试时，模型在与编辑无关的输入上仍然输出目标对象 o* 的比例。

    简化实现：
      - 认为 fact_A_after.answer 就是编辑后的目标对象 o*；
      - 在 locality prompt B 上生成的 response 如果包含该对象字符串，
        则视作一次「over-editing」。
    """
    if not results:
        return 0.0

    over_count = 0.0
    total = 0.0
    for r in results:
        target = (r.fact_A_after.answer or "").strip().lower()
        resp_B = (r.fact_B_after.response or "").strip().lower()
        if target and resp_B:
            if target in resp_B:
                over_count += 1.0
        total += 1.0

    return over_count / total if total > 0 else 0.0


# ====== Fluency：n-gram 熵 ======

def _ngram_entropy(texts: List[str], n: int) -> float:
    """
    计算 n-gram 熵：
      H_n = - sum_k f(k) log_2 f(k),
    其中 f(·) 是 n-gram 频率分布（频率 / 总 n-gram 数）。

    对应论文里 Fluency 的定义（使用二元/三元 n-gram 频率）。:contentReference[oaicite:2]{index=2}
    """
    counts: Counter = Counter()
    total = 0

    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            counts[ngram] += 1
            total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy += -p * math.log2(p)

    return entropy


def compute_fluency(
    texts: List[str],
    w_bigram: float = 0.5,
    w_trigram: float = 0.5,
) -> float:
    """
    Fluency 指标：二元和三元 n-gram 熵的加权和。

      Fluency = w2 * H_2 + w3 * H_3
    """
    if not texts:
        return 0.0
    h2 = _ngram_entropy(texts, 2)
    h3 = _ngram_entropy(texts, 3)
    return w_bigram * h2 + w_trigram * h3


# ====== 汇总一组 EditPairResult，用于 run_experiment.summarize_edit_pairs ======

def summarize_edit_pairs(results: List[EditPairResult]) -> Dict[str, float]:
    """
    对一组 EditPairResult 按 FiNE / KnowEdit 指标做简单汇总。

    输出：
      - edit_success: Edit Success
      - locality: Locality
      - over_editing_rate: 过度编辑率
      - A_acc_before / A_acc_after: 被编辑事实的编辑前后准确率
      - B_acc_before / B_acc_after: locality 事实的编辑前后准确率
      - fluency_A_after / fluency_B_after: 基于生成文本的 Fluency（可选）
    """
    if not results:
        return {}

    n = len(results)

    # A: 被编辑事实
    A_acc_before = sum(1.0 for r in results if r.fact_A_before.is_correct) / n
    A_acc_after = sum(1.0 for r in results if r.fact_A_after.is_correct) / n

    # B: locality 测试事实
    B_acc_before = sum(1.0 for r in results if r.fact_B_before.is_correct) / n
    B_acc_after = sum(1.0 for r in results if r.fact_B_after.is_correct) / n

    edit_success = compute_edit_success(results)
    locality = compute_locality(results)
    over_editing_rate = compute_over_editing_rate(results)

    # Fluency：对 A_after / B_after 的生成文本分别统计
    texts_A_after = [
        r.fact_A_after.response
        for r in results
        if r.fact_A_after.response is not None
    ]
    texts_B_after = [
        r.fact_B_after.response
        for r in results
        if r.fact_B_after.response is not None
    ]
    fluency_A_after = compute_fluency(texts_A_after) if texts_A_after else 0.0
    fluency_B_after = compute_fluency(texts_B_after) if texts_B_after else 0.0

    return {
        "num_pairs": float(n),
        "edit_success": edit_success,
        "A_acc_before": A_acc_before,
        "A_acc_after": A_acc_after,
        "B_acc_before": B_acc_before,
        "B_acc_after": B_acc_after,
        "locality": locality,
        "over_editing_rate": over_editing_rate,
        "fluency_A_after": fluency_A_after,
        "fluency_B_after": fluency_B_after,
    }


def pretty_print_summary(title: str, summary: Dict[str, float]) -> None:
    """
    简单打印 summary。run_experiment 会用到。
    """
    print(f"\n===== {title} =====")
    if not summary:
        print("No results.")
        return

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:8.4f}")
        else:
            print(f"{k:20s}: {v}")
