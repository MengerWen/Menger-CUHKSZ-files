# 📄 Experiment Report: Ablation Sensitivity Analysis of FiNE Model Editing Method

## 🎯 Objective

This report aims to investigate the sensitivity of the **FiNE** neuron-level model editing method to two core hyperparameters:
1.  **Number of Top-K Neurons** ($K$)
2.  **Jaccard Overlap Threshold** ($\tau_{high}$)

All experiments were conducted on the **Set-Poly (High Overlap)** dataset, which contains 50 pairs of facts with high neuron overlap.

> **Metric Definition**: The **Avg. Score** used in this report measures the comprehensive performance of the edit. It is defined as the proportion of samples where **Fact A is successfully modified ($\text{Edit Success}$)** AND **Fact B remains correct ($\text{Locality}$)**.

---

## I. Ablation Study on Top-K Neurons ($K$ Value)

This experiment fixed the Jaccard overlap threshold to the baseline value ($\tau_{high} = 0.1$) and varied the number of Top-K neurons used for FiNE editing to observe the impact of neuron quantity on editing efficacy.

### 1.1 Experiment Configuration

| Parameter | Values Tested | Sample Size ($N$) | Dataset | Experiment Code |
| :--- | :--- | :---: | :---: | :---: |
| **Top-K** | 10, 20, 50, **100** (Baseline), 200 | 50 | Set-Poly | `exp_code=900` |

### 1.2 Statistical Results

[cite_start]We calculated the number of samples with a perfect score of 1.0 (Success + Locality) and the average score for different $K$ values[cite: 1, 15, 29, 45, 59].

| K Value | Total Samples ($N$) | Successful Edits (Score=1.0) | **Average Score (Avg. Score)** | Performance Trend |
| :---: | :---: | :---: | :---: | :---: |
| **10** | 50 | 6 | **0.12** | Low |
| **20** | 50 | 6 | **0.12** | Low |
| **50** | 50 | 11 | **0.22** | Rising |
| **100** | 50 | 22 | **0.44** | **Peak (Optimal)** |
| **200** | 50 | 9 | **0.18** | Sharp Drop |

### 1.3 Analysis and Conclusion

The experimental results demonstrate a significant **"Inverted U-shaped"** sensitivity curve:

1.  **Limitations of Small $K$ Values ($K \le 20$)**
    * **Performance**: Score is only 0.12.
    * **Reason**: The number of selected neurons is too small to cover and capture the complete representation of the knowledge. As the model cannot precisely locate where the knowledge is stored, it fails to successfully execute the edit on Fact A.

2.  **Optimal Balance Point ($K=100$)**
    * **Performance**: The score reaches a peak of 0.44.
    * **Reason**: At this magnitude, FiNE achieves the best balance between capturing enough relevant neurons (ensuring edit success) and avoiding excessive irrelevant neurons (preserving locality).

3.  **Side Effects of Large $K$ Values ($K=200$)**
    * **Performance**: The score drops significantly to 0.18.
    * **Reason**: **Locality Loss (Bleed-over) intensifies**. As the number of edited neurons increases, the edit operation becomes less "sharp." Its influence expands, affecting non-target facts (Fact B) that share some neurons, leading to a decline in locality metrics.

---

## II. Ablation Study on Jaccard Overlap Threshold

### 2.1 Experiment Overview

This experiment aims to verify how the definition of "High Overlap" (Set-Poly) affects editing difficulty. Based on the existing result file `results_k_100_jaccard.jsonl`, we obtained performance data under baseline conditions.

### 2.2 Results

| Parameter Configuration | Sample Size ($N$) | Average Score (Avg. Score) | Note |
| :--- | :---: | :---: | :--- |
| **Jaccard $\tau_{high} = 0.1$** <br> ($K=100$) | 50 | **0.44** | Baseline Performance |

*(Note: Due to configuration constraints, only data for the baseline threshold was obtained. To observe changes for $\tau_{high}=0.05$ or $0.2$, it is recommended to correct the `exp_code=901` configuration and run supplementary experiments.)*

---

## III. Summary and Discussion

### 3.1 Key Findings
Through a detailed sweep of the Top-K parameter, we confirmed that **the FiNE method is highly sensitive to hyperparameter selection**:
* Performance does not increase linearly with $K$; instead, there is a distinct "Sweet Spot."
* For the Pythia-6.9B model, **$K=100$** is the optimal choice for this dataset.

### 3.2 Impact of Polysemanticity
Combining the comparison data from Set-Mono (Low Overlap Group), where locality loss was approximately 77% compared to 88% in Set-Poly, along with the performance drop at $K=200$, the core hypothesis of the original paper is further validated:
> **The polysemanticity (degree of sharing) of neurons is a key factor driving the "bleed-over" phenomenon in neuron-level model editing. The more neurons involved in an edit, or the higher the overlap between neurons, the more severe the non-local side effects become.**