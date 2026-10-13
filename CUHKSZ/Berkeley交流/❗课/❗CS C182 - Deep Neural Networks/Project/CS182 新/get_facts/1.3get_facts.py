import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ==========================================
# 1. 核心计算类 (逻辑保持不变)
# ==========================================
class NeuronOverlapCalculator:
    def __init__(self, model_name="EleutherAI/pythia-160m", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] 正在加载模型 {model_name} 到 {self.device}...")
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # 预先获取 Unembedding 矩阵 (Vocab x Hidden)
        # 注意: Pythia/GPT-NeoX 的输出层通常是 embed_out
        self.W_u = self.model.embed_out.weight.detach()

    def get_important_neurons(self, prompt, target, layer_indices, top_k=50):
        """
        计算给定 Prompt + Target 下，Target 部分每一个 Token 对应的关键神经元集合。
        逻辑: FiNE Score (Act * Projected_Weight) -> Global Ranking -> Union across tokens
        """
        
        # --- 数据准备与 Token index 对齐 ---
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        full_text = prompt + target
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        full_ids = inputs["input_ids"]
        
        seq_len = full_ids.shape[1]
        prompt_len = prompt_ids.shape[1]
        
        # Target tokens 的范围：[prompt_len, seq_len)
        target_token_indices = range(prompt_len, seq_len)
        
        if len(target_token_indices) == 0:
            # 这种情况通常发生在 target 为空或者分词后没有产生新 token
            return set()

        # --- 注册 Hook 获取全序列激活值 ---
        activations = {}
        handles = []
        
        def get_act_hook(layer_idx):
            def hook(module, input, output):
                # Pythia MLP: dense_h_to_4h -> act -> dense_4h_to_h
                activations[layer_idx] = output.detach()
            return hook

        for layer_idx in layer_indices:
            h = self.model.gpt_neox.layers[layer_idx].mlp.act.register_forward_hook(get_act_hook(layer_idx))
            handles.append(h)

        # --- 前向传播 ---
        with torch.no_grad():
            self.model(**inputs)
            
        for h in handles: h.remove()

        # --- 逐 Token 计算分数并进行 Global Ranking ---
        final_neuron_set = set()

        # 遍历 Target 中的每一个 Token
        for t_idx in target_token_indices:
            target_id = full_ids[0, t_idx].item()
            act_idx = t_idx - 1 # 预测 t_idx 需要的是 t_idx-1 的激活
            
            # 获取 Unembedding 向量 [1, Hidden]
            w_u_vec = self.W_u[target_id, :].unsqueeze(0)
            
            all_layer_scores_list = []

            for layer_idx in layer_indices:
                # 获取激活值 [Intermediate]
                q = activations[layer_idx][0, act_idx, :]
                
                # 获取 Output Weight [Hidden, Intermediate]
                W_out = self.model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight 
                
                # 计算投影: (W_u * W_out) -> [1, Intermediate]
                projected_w = torch.matmul(w_u_vec, W_out).squeeze()
                
                # 计算分数: |Activation * Projected_Weight|
                layer_scores = torch.abs(q * projected_w)
                
                # 性能优化: 先在层内取 Top-k，再做全局排序
                vals, idxs = torch.topk(layer_scores, k=top_k)
                for v, i in zip(vals, idxs):
                    all_layer_scores_list.append((v.item(), layer_idx, i.item()))
            
            # Global Ranking across all layers
            all_layer_scores_list.sort(key=lambda x: x[0], reverse=True)
            
            # 取全局 Top-k
            current_token_top_neurons = all_layer_scores_list[:top_k]
            
            # 加入集合
            for _, l, n_idx in current_token_top_neurons:
                final_neuron_set.add((l, n_idx))

        return final_neuron_set

    def compute_jaccard(self, fact_A, fact_B, layers, top_k=50):
        # 1. Fact A 关键神经元
        set_A = self.get_important_neurons(
            fact_A["prompt"], fact_A["target"], layers, top_k
        )
        
        # 2. Fact B 关键神经元
        set_B = self.get_important_neurons(
            fact_B["prompt"], fact_B["target"], layers, top_k
        )
        
        # 3. 计算 Jaccard
        if len(set_A) == 0 or len(set_B) == 0:
            return 0.0
            
        intersection = len(set_A.intersection(set_B))
        union = len(set_A.union(set_B))
        
        return intersection / union

# ==========================================
# 2. 主程序：批量处理与可视化
# ==========================================
def main():
    # --- 配置参数 ---
    json_file = "fact_pairs.json"
    output_file = "jaccard_results.json"
    model_name = "EleutherAI/pythia-160m"
    target_layers = [5, 6, 7, 8, 9]  # 针对 160m 模型选择中间层
    TOP_K = 50

    # 1. 检查数据是否存在
    if not os.path.exists(json_file):
        print(f"[Error] 找不到输入文件: {json_file}")
        print("请确保你已经在当前目录下创建了 'fact_pairs.json' 文件。")
        return

    # 2. 加载模型
    calculator = NeuronOverlapCalculator(model_name=model_name)

    # 3. 读取数据
    print(f"[Info] 正在读取 {json_file} ...")
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"\n[Batch] 开始处理 {len(dataset)} 组数据...")
    print(f"        Layers: {target_layers} | Top-K: {TOP_K}")

    results = []
    group_scores = {} # 用于绘图分类

    # 4. 循环计算
    for i, item in enumerate(dataset):
        group = item.get("group", "Unknown")
        fact_A = item["fact_A"]
        fact_B = item["fact_B"]

        try:
            score = calculator.compute_jaccard(fact_A, fact_B, target_layers, top_k=TOP_K)
            
            # 记录结果
            record = {
                "id": i,
                "group": group,
                "jaccard": score,
                "target_A": fact_A["target"],
                "target_B": fact_B["target"]
            }
            results.append(record)

            # 收集分组数据以便绘图
            if group not in group_scores:
                group_scores[group] = []
            group_scores[group].append(score)

            # 打印进度
            if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
                print(f"Processing {i+1}/{len(dataset)} | Last Jaccard: {score:.4f} ({group})")

        except Exception as e:
            print(f"[Error] Item {i} failed: {e}")

    # 5. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] 详细结果已保存至: {output_file}")

    # 6. 可视化 (Boxplot)
    if len(results) > 0:
        print("[Plot] 正在绘制分布图...")
        try:
            plt.figure(figsize=(10, 6))
            
            # 准备绘图数据 (排序保证Label顺序一致)
            labels = sorted(group_scores.keys())
            data_to_plot = [group_scores[k] for k in labels]
            
            # 绘制箱线图
            plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            plt.title(f'Neuron Overlap Distribution (FiNE, Top-K={TOP_K})')
            plt.ylabel('Jaccard Similarity')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 保存图片
            plot_filename = "overlap_distribution.png"
            plt.savefig(plot_filename)
            print(f"[Done] 图表已保存至: {plot_filename}")
            # plt.show() # 如果在 Jupyter 中运行可以取消注释
            
        except Exception as e:
            print(f"[Warning] 绘图失败 (可能是缺少图形库): {e}")
    else:
        print("[Warning] 没有生成有效结果，跳过绘图。")

if __name__ == "__main__":
    main()