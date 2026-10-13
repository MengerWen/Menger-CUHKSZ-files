# Experiments to Find High Overlap Facts (A & B)

To find a pair with high Jaccard similarity, we should test the following categories.

## 1. Candidate Pairs

### Category A: Shared Subject (Most Promising)
*Hypothesis: The model activates "Subject Neurons" (e.g., representing "Steve Jobs") to retrieve different attributes.*

*   **Pair 1 (Steve Jobs):**
    *   **Fact A:** `prompt="Steve Jobs founded"`, `target=" Apple"`
    *   **Fact B:** `prompt="Steve Jobs worked at"`, `target=" Pixar"`
*   **Pair 2 (Bill Gates):**
    *   **Fact A:** `prompt="Bill Gates founded"`, `target=" Microsoft"`
    *   **Fact B:** `prompt="Bill Gates is famous for"`, `target=" Windows"`

### Category B: Shared Output/Object
*Hypothesis: The model activates "Concept Neurons" related to the target word.*

*   **Pair 3 (Paris):**
    *   **Fact A:** `prompt="The capital of France is"`, `target=" Paris"`
    *   **Fact B:** `prompt="The Eiffel Tower is located in"`, `target=" Paris"`
*   **Pair 4 (Language):**
    *   **Fact A:** `prompt="Hello in Spanish is"`, `target=" Hola"`
    *   **Fact B:** `prompt="Hola means Hello in"`, `target=" Spanish"` (Note: Target differs, check cross-activation)

## 2. Parameter Tuning Strategy

### Model
*   **Start with:** `EleutherAI/pythia-160m` (Fast, good for debugging)
*   **Upgrade to:** `EleutherAI/pythia-410m` or `gpt2-medium` if overlap is too noisy.

### Layers to Scan
Knowledge is usually located in the **middle-to-late MLP layers**.
*   **For 12-layer model (160m):** Scan layers `range(5, 10)` (Layers 5, 6, 7, 8, 9)
*   **For 24-layer model (410m+):** Scan layers `range(10, 20)`

### Top-K Neurons
This is the most sensitive parameter.
*   **Start with:** `k=50`
*   **If Jaccard < 0.05:** Increase `k` to `100` or `200`. (Small models distribute info broadly)
*   **If Jaccard > 0.3:** Decrease `k` to `20` to see if the overlap is truly "core" neurons.

## 3. Recording Results (Template)

| Pair ID | Model | Layers | Top-K | Jaccard Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pair 1 (Jobs) | 160m | 5-10 | 50 | 0.0593 | Good Candidate |
| Pair 1 (Jobs) | 160m | 5-10 | 100 | 0.0941 | Good Candidate |
| Pair 2 (Gates) | 160m | 5-10 | 50 | 0.0707 | Good Candidate |
| Pair 2 (Gates) | 160m | 5-10 | 100 | 0.1148 | High Overlap! ⭐ |
| Pair 3 (Paris) | 160m | 5-10 | 50 | 0.0870 | Good Candidate |
| Pair 3 (Paris) | 160m | 5-10 | 100 | 0.1236 | High Overlap! ⭐ |
| Pair 4 (Language) | 160m | 5-10 | 50 | 0.0526 | Good Candidate |
| Pair 4 (Language) | 160m | 5-10 | 100 | 0.1038 | High Overlap! ⭐ |
