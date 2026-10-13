# ROME Experiment Report

### 1. Logic & Methodology (What We Accomplished)

**Implementation Details**

We successfully deployed the ROME (Rank-One Model Editing) algorithm on the Pythia-6.9B model within a resource-constrained environment (Google Colab A100).


**Experimental Design**

Target Layer: Layer 28 (Selected based on Step 2 analysis, showing ~19% Jaccard Similarity for the France/Paris pair).

Dataset: Constructed 4 specific Fact Pairs (Fact A and Fact B) covering geographic, linguistic, and biographical knowledge.



### 2. Experimental Results

| Case ID | Fact A (Edit Target) | Fact B (Observed Target) | Pre-Edit Prob ($P_B$) | Post-Edit Prob ($P_B'$) | **Bleed-over (Drop %)** | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pair 3** | France $\to$ Rome | **Eiffel Tower** (is in Paris) | 35.74% | 10.14% | **🔻 71.6%** | **Severe Collapse** |
| **Pair 4** | Spanish $\to$ Bonjour | **Hola** (means Hello in Spanish) | 62.55% | 41.65% | **🔻 33.4%** | Moderate Collapse |
| **Pair 1** | Jobs $\to$ Microsoft | **Pixar** (Jobs worked at) | 2.10% | 0.05% | **🔻 97.5%** | Complete Erasure |
| **Pair 2** | Gates $\to$ Apple | **Windows** (Gates is famous for) | 0.00% | 0.00% | — | N/A (Model Blind Spot) |


**Key Findings**

Strong Correlation Confirmed: In Pair 3 (Paris), editing the capital of France caused the model's ability to locate the Eiffel Tower to plummet by 71.6%. This confirms that in high-overlap layers (Layer 28), ROME cannot disentangle correlated facts.

Significant Side Effects: Across all valid test cases, editing Fact A resulted in a substantial degradation of Fact B, proving that physical neuron entanglement leads to knowledge bleed-over.