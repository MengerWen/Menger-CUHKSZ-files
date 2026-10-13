import torch
import json
import os
import gc
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig

# ================= 配置区域 =================
INPUT_FILE = "fact_pairs.json"
OUTPUT_FILE = "all_models_similarity_results.json"

# 定义你要跑的四个模型 (A, B, C, D)
MODELS_CONFIG = {
    "Pythia-70m": "EleutherAI/pythia-70m",
    "Pythia-160m": "EleutherAI/pythia-160m",
    "Pythia-410m": "EleutherAI/pythia-410m",
    "Pythia-1b": "EleutherAI/pythia-1b"
}

TOP_K = 50

# ==========================================
# 1. 核心计算类 (严格保持你的逻辑不变)
# ==========================================
class NeuronOverlapCalculator:
    def __init__(self, model_name, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] 正在加载模型 {model_name} 到 {self.device}...")
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        # 预先获取 Unembedding 矩阵
        self.W_u = self.model.embed_out.weight.detach()

    def get_important_neurons(self, prompt, target, layer_indices, top_k=50):
        # --- 数据准备与 Token index 对齐 ---
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        full_text = prompt + target
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        full_ids = inputs["input_ids"]

        seq_len = full_ids.shape[1]
        prompt_len = prompt_ids.shape[1]
        target_token_indices = range(prompt_len, seq_len)

        if len(target_token_indices) == 0: return set()

        # --- 注册 Hook ---
        activations = {}
        handles = []
        def get_act_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output.detach()
            return hook

        for layer_idx in layer_indices:
            h = self.model.gpt_neox.layers[layer_idx].mlp.act.register_forward_hook(get_act_hook(layer_idx))
            handles.append(h)

        # --- 前向传播 ---
        with torch.no_grad():
            self.model(**inputs)
        for h in handles: h.remove()

        # --- 逐 Token 计算 ---
        final_neuron_set = set()
        for t_idx in target_token_indices:
            target_id = full_ids[0, t_idx].item()
            act_idx = t_idx - 1 
            w_u_vec = self.W_u[target_id, :].unsqueeze(0)
            all_layer_scores_list = []

            for layer_idx in layer_indices:
                q = activations[layer_idx][0, act_idx, :]
                W_out = self.model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight
                projected_w = torch.matmul(w_u_vec, W_out).squeeze()
                layer_scores = torch.abs(q * projected_w)
                
                vals, idxs = torch.topk(layer_scores, k=top_k)
                for v, i in zip(vals, idxs):
                    all_layer_scores_list.append((v.item(), layer_idx, i.item()))

            all_layer_scores_list.sort(key=lambda x: x[0], reverse=True)
            current_token_top_neurons = all_layer_scores_list[:top_k]

            for _, l, n_idx in current_token_top_neurons:
                final_neuron_set.add((l, n_idx))

        return final_neuron_set

    def compute_jaccard(self, fact_A, fact_B, layers, top_k=50):
        set_A = self.get_important_neurons(fact_A["prompt"], fact_A["target"], layers, top_k)
        set_B = self.get_important_neurons(fact_B["prompt"], fact_B["target"], layers, top_k)
        if len(set_A) == 0 or len(set_B) == 0: return 0.0
        intersection = len(set_A.intersection(set_B))
        union = len(set_A.union(set_B))
        return intersection / union

# ==========================================
# 2. 主程序：循环模型并汇总
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 初始化汇总数据结构
    # 结构: [{"id": 0, "group": "High", "similarity_Pythia-70m": 0.1, ...}, ...]
    aggregated_results = []
    for i, item in enumerate(raw_data):
        aggregated_results.append({
            "id": i,
            "group": item.get("group", "Unknown"),
            "fact_A_target": item["fact_A"]["target"],
            "fact_B_target": item["fact_B"]["target"]
        })

    # 循环处理每个模型
    for short_name, full_model_name in MODELS_CONFIG.items():
        print(f"\n{'='*40}")
        print(f"开始处理模型: {short_name} ({full_model_name})")
        print(f"{'='*40}")

        # 1. 动态确定层数 (Middle Layers)
        # 必须这样做，因为 70m 只有 6 层，1b 有 16 层
        config = AutoConfig.from_pretrained(full_model_name)
        n_layers = config.num_hidden_layers
        start_layer = int(n_layers * 0.33)
        end_layer = int(n_layers * 0.66)
        target_layers = list(range(start_layer, end_layer))
        print(f"-> 模型总层数: {n_layers}, 选取扫描层: {target_layers}")

        # 2. 加载模型
        calculator = NeuronOverlapCalculator(model_name=full_model_name)

        # 3. 计算该模型下的所有 Jaccard
        for idx, item in enumerate(raw_data):
            try:
                score = calculator.compute_jaccard(
                    item["fact_A"], 
                    item["fact_B"], 
                    layers=target_layers, 
                    top_k=TOP_K
                )
                
                # 将结果写入对应的 ID 中，key 为 "similarity_模型名"
                aggregated_results[idx][f"similarity_{short_name}"] = score
                
            except Exception as e:
                print(f"Error computing item {idx}: {e}")
                aggregated_results[idx][f"similarity_{short_name}"] = 0.0

        # 4. 清理显存 (防止 OOM)
        print(f"-> 清理 {short_name} 显存...")
        del calculator
        del config
        torch.cuda.empty_cache()
        gc.collect()

    # 保存最终汇总文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[全部完成] 所有模型数据已汇总至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()