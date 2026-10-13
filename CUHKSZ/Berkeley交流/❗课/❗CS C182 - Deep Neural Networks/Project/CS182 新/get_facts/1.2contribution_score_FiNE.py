import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

def get_gradient_free_contribution_scores(model, tokenizer, prompt, target, layer_indices):
    """
    基于 ACL 2024 "Finding and Editing Multi-Modal Neurons" 论文实现。
    计算方法: Activation * (Output_Weight · Unembedding_Weight)
    特点: 无需梯度 (Gradient-free)，纯前向计算 + 权重投影。
    
    参数:
        model: Pythia (GPTNeoX) 模型
        tokenizer: 分词器
        prompt: 输入文本 (例如 "The capital of France is")
        target: 目标词 (例如 " Paris")
        layer_indices: 要计算的层 (例如 range(10, 20))
    """
    
    # 1. 准备输入
    # 只需要 Prompt 的前向传播来获取激活值
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 获取 Target Token 的 ID (用于从 Unembedding 矩阵中提取向量)
    # 注意：Target 最好是一个单词，确保它对应词表里的一个 Token
    target_token_id = tokenizer.encode(target)[0]
    print(f"Calculating contribution for target token: '{target}' (ID: {target_token_id})")

    # 2. 准备钩子 (Hooks) 获取激活值
    # 论文中的 o^l_i (Activation)
    activations = {}
    handles = []

    def get_activation_hook(layer_idx):
        def hook(module, args, output):
            # 对于 Pythia 的 dense_4h_to_h 层：
            # args[0] 是输入，即 dense_h_to_4h + GeLU 之后的激活值
            # 这正是我们需要捕获的 "FFN 中间层神经元激活"
            # Shape: (Batch, Seq, Intermediate_Dim)
            
            # 我们只取 Last Token ([:, -1, :])
            activations[layer_idx] = args[0][:, -1, :].detach()
        return hook

    # 3. 注册钩子
    # Pythia 结构: gpt_neox.layers[i].mlp.dense_4h_to_h
    for i in layer_indices:
        layer_module = model.gpt_neox.layers[i].mlp.dense_4h_to_h
        # 注意：我们需要的是这个线性层的“输入”，因为它代表了神经元的激活
        handles.append(layer_module.register_forward_hook(get_activation_hook(i)))

    # 4. 运行前向传播 (No Gradient Needed!)
    with torch.no_grad():
        model(**inputs)

    # 5. 获取静态权重矩阵
    # (1) Unembedding Matrix (W_u): 负责将隐状态映射到词表
    # Pythia 中通常叫 embed_out
    w_u = model.embed_out.weight # Shape: (Vocab, Hidden)
    
    # 提取目标 Token 对应的向量 v (论文中的 v)
    # Shape: (Hidden,)
    target_vector = w_u[target_token_id, :] 

    scores = {}
    
    # 6. 计算 Contribution Score
    print("Computing scores using Weight Projection...")
    
    for i in layer_indices:
        # --- A. 获取激活值 (o) ---
        # Shape: (Intermediate_Dim,) e.g., 4096*4
        act = activations[i].squeeze() 
        
        # --- B. 获取 FFN 输出权重 (W_out) ---
        # Pythia 的 dense_4h_to_h.weight 形状通常是 (Hidden, Intermediate)
        # 这是一个从 4h 投影回 h 的矩阵
        # 其中的第 j 列，代表第 j 个神经元对 Residual Stream 的贡献方向
        w_out = model.gpt_neox.layers[i].mlp.dense_4h_to_h.weight
        
        # --- C. 投影计算 (Projected Weight) ---
        # 论文逻辑: W_out 的列向量 与 Target_Vector 的点积
        # 这一步计算：每个神经元的输出方向，是否与目标 Token 的向量方向一致？
        # 数学操作: vector (H) * matrix (H, 4H) -> vector (4H)
        # 也就是：Target_Vector @ W_out
        
        # projection[j] = W_out[:, j] · Target_Vector
        # Shape: (Intermediate_Dim,)
        neuron_projection = torch.matmul(target_vector, w_out)
        
        # --- D. 最终分数 (Score) ---
        # Score = Activation * Projection
        # 逐元素相乘 (Element-wise multiplication)
        # 只有当神经元“激活了”且“方向对”，分数才高
        layer_scores = act * neuron_projection
        
        # 存入结果 (保留原始的正负号，也可以取 abs)
        # 论文中通常取原始值来看正负贡献，但为了找“关键神经元”，取 abs 也可以
        # 这里我们按照论文 Algorithm 1 的逻辑，先计算原始值
        scores[i] = layer_scores

    # 清理钩子
    for h in handles:
        h.remove()
        
    return scores

# ==========================================
# 0. 准备工作：加载模型
# ==========================================
print("正在加载模型...")
model_name = "EleutherAI/pythia-160m" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和分词器
model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型加载完毕: {model_name}, 设备: {device}")

# ==========================================
# 3. 执行实验：批量测试 Fact Pairs (FiNE Method)
# ==========================================

# 设定我们要扫描的层
check_layers = range(5, 10) 
top_k_list = [50, 100] 

# 定义我们要测试的事实对
fact_pairs = [
    {
        "id": "Pair 1 (Jobs)",
        "fact_A": {"prompt": "Steve Jobs founded", "target": " Apple"},
        "fact_B": {"prompt": "Steve Jobs worked at", "target": " Pixar"}
    },
    {
        "id": "Pair 2 (Gates)",
        "fact_A": {"prompt": "Bill Gates founded", "target": " Microsoft"},
        "fact_B": {"prompt": "Bill Gates is famous for", "target": " Windows"}
    },
    {
        "id": "Pair 3 (Paris)",
        "fact_A": {"prompt": "The capital of France is", "target": " Paris"},
        "fact_B": {"prompt": "The Eiffel Tower is located in", "target": " Paris"}
    },
    {
        "id": "Pair 4 (Language)",
        "fact_A": {"prompt": "Hello in Spanish is", "target": " Hola"},
        "fact_B": {"prompt": "Hola means Hello in", "target": " Spanish"}
    }
]

print(f"\n=== 开始批量实验 (FiNE Method) ===")
print(f"{'Pair ID':<20} | {'Top-K':<6} | {'Jaccard':<8} | {'Overlap Count':<13} | {'Notes'}")
print("-" * 80)

for pair in fact_pairs:
    # 1. 计算 Fact A 的分数 (使用 FiNE 方法)
    scores_A = get_gradient_free_contribution_scores(
        model, tokenizer, 
        prompt=pair["fact_A"]["prompt"], 
        target=pair["fact_A"]["target"], 
        layer_indices=check_layers
    )
    
    # 2. 计算 Fact B 的分数 (使用 FiNE 方法)
    scores_B = get_gradient_free_contribution_scores(
        model, tokenizer, 
        prompt=pair["fact_B"]["prompt"], 
        target=pair["fact_B"]["target"], 
        layer_indices=check_layers
    )
    
    # 3. 针对不同的 Top-K 计算重叠度
    for k in top_k_list:
        set_A = set()
        set_B = set()

        for layer in check_layers:
            # 处理 Fact A (取绝对值后排序)
            top_indices_A = torch.topk(torch.abs(scores_A[layer]), k).indices.cpu().numpy()
            for idx in top_indices_A:
                set_A.add( (layer, idx) ) 
                
            # 处理 Fact B
            top_indices_B = torch.topk(torch.abs(scores_B[layer]), k).indices.cpu().numpy()
            for idx in top_indices_B:
                set_B.add( (layer, idx) )

        # 计算 Jaccard Overlap
        intersection = set_A.intersection(set_B)
        union = set_A.union(set_B)
        
        if len(union) > 0:
            jaccard_index = len(intersection) / len(union)
        else:
            jaccard_index = 0.0
            
        # 简单的评价
        note = ""
        if jaccard_index > 0.1: note = "High Overlap! ⭐"
        elif jaccard_index > 0.05: note = "Good Candidate"
        
        print(f"{pair['id']:<20} | {k:<6} | {jaccard_index:.4f}   | {len(intersection):<13} | {note}")

print("-" * 80)
print("FiNE 方法实验结束。")