import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ==========================================
# 1. 准备工作：加载模型
# ==========================================
print("正在加载模型...")
# 如果显存不够，把 "EleutherAI/pythia-6.9b" 换成 "EleutherAI/pythia-160m" 先跑通逻辑
model_name = "EleutherAI/pythia-160m" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和分词器
model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型加载完毕: {model_name}, 设备: {device}")

# ==========================================
# 2. 定义核心工具：获取神经元分数的函数
# ==========================================
def get_neuron_scores(model, prompt, target_word, layer_indices):
    
    """
    输入:
        prompt: 提示词 (例如 "The capital of France is")
        target_word: 目标词 (例如 " Paris") 注意前面可能有空格
        layer_indices: 我们想检查哪些层 (例如 [10, 11, 12])
    输出:
        scores: 一个字典，key是层号，value是该层每个神经元的分数
    """
    
    # --- 步骤 A: 准备输入数据 ---
    input_text = prompt + target_word
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 找到目标词 (target_word) 对应的 Token ID
    # 这是一个简化处理，假设 target 只是最后一个 token
    target_ids = inputs["input_ids"].clone()
    
    # --- 步骤 B: 设置“钩子” (Hooks) ---
    # 钩子就像是我们在电路板上接的电表，用来记录电流(激活)和电压变化(梯度)
    activations = {} 
    gradients = {}
    handles = [] # 用来存钩子的句柄，用完要拆掉

    # 定义：怎么记录激活值
    def get_act_hook(layer_idx):
        def hook(module, input, output):
            # output 是这一层的输出。
            # 【修正】我们需要看倒数第二个 token (index -2)，因为它负责预测最后一个 token (Target)
            # 如果取 -1，那是最后一个 token 用来预测“未来”的，梯度通常为 0
            activations[layer_idx] = output[:, -2, :].detach()
        return hook

    # 定义：怎么记录梯度
    def get_grad_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output 是反向传回来的梯度
            gradients[layer_idx] = grad_output[0][:, -2, :].detach()
        return hook

    # 把钩子挂到模型上
    # Pythia 的结构里，MLP 的核心输出层叫 dense_4h_to_h
    for layer_idx in layer_indices:
        # 这一行代码是在找模型里的具体某一层
        target_layer = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h
        
        # 挂上钩子
        handles.append(target_layer.register_forward_hook(get_act_hook(layer_idx)))
        handles.append(target_layer.register_full_backward_hook(get_grad_hook(layer_idx)))

    # --- 步骤 C: 运行模型 (Forward & Backward) ---
    model.zero_grad() # 清空之前的梯度
    
    # 1. 前向传播：算 Loss
    # 我们告诉模型："输入是 inputs，标准答案是 target_ids，请算算你错得离谱吗？(Loss)"
    outputs = model(**inputs, labels=target_ids)
    loss = outputs.loss
    
    # 2. 反向传播：算梯度
    # 模型会从 Loss 开始往回推，算出每个神经元对 Loss 的责任大小
    loss.backward()
    
    # --- 步骤 D: 计算分数 ---
    scores = {}
    for layer_idx in layer_indices:
        act = activations[layer_idx].squeeze() # 拿到激活值
        grad = gradients[layer_idx].squeeze()  # 拿到梯度
        
        # 核心公式：Score = | Activation * Gradient |
        # torch.abs 是取绝对值
        score = torch.abs(act * grad)
        scores[layer_idx] = score

    # 拆掉钩子 (打扫战场)
    for h in handles:
        h.remove()
        
    return scores

# 另一种评分方法：直接用投影法 (不需要梯度) 就是FiNE论文中用的方法
def get_fine_scores(model, prompt, target_word, layer_indices):
    # 1. 拿到 Unembedding Matrix (W_u)
    # Pythia 的 embed_out 通常就是 embed_in 的转置，或者是专门的层
    W_u = model.embed_out.weight # 形状通常是 [vocab_size, hidden_dim]
    
    # 2. 准备输入
    input_text = prompt + target_word
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # 获取 target token id
    target_id = inputs["input_ids"][0, -1].item() 

    # 3. 获取激活值 (只需要 Forward Hook)
    activations = {}
    def get_act_hook(layer_idx):
        def hook(module, input, output):
            # 这里的 output 形状是 [batch, seq_len, hidden_dim] (或者是 MLP 中间层大小)
            # 注意：Pythia MLP 是 up_proj -> act -> down_proj
            # 截图里的 q 是 "neurons in FFN"，通常指激活函数后的值
            # 但我们需要的是 MLP 具体的“神经元”。
            # 在 Pythia 中，mlp.dense_h_to_4h 之后是激活，这里是中间层
            activations[layer_idx] = output[:, -2, :].detach() # 取倒数第二个token
        return hook

    handles = []
    for layer_idx in layer_indices:
        # 挂载到激活函数之后，down_projection 之前
        # Pythia: layers[i].mlp.act
        target_layer = model.gpt_neox.layers[layer_idx].mlp.act
        handles.append(target_layer.register_forward_hook(get_act_hook(layer_idx)))

    # 4. 前向传播
    with torch.no_grad():
        model(**inputs)

    # 5. 计算分数 (公式 3)
    scores = {}
    for layer_idx in layer_indices:
        # q: [intermediate_size] (神经元激活值)
        q = activations[layer_idx].squeeze() 
        
        # W_out: [hidden_dim, intermediate_size] (MLP 的输出/向下投影矩阵)
        # Pythia: layers[i].mlp.dense_4h_to_h.weight
        W_out = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight.T 
        
        # W_u: [vocab_size, hidden_dim]
        # 我们只关心 target token 对应的那一行 W_u_target: [1, hidden_dim]
        W_u_target = W_u[target_id, :] 
        
        # 计算投影 (Projected Weight)
        # 公式: (W_u * W_out) 
        # 我们可以先算 W_u_target * W_out，得到一个 [intermediate_size] 的向量
        # 这个向量代表：每个中间层神经元对 Target Token 的直接贡献系数
        projected_weight = torch.matmul(W_u_target, W_out) 
        
        # 最终分数: 激活值 * 投影系数
        score = q * projected_weight
        scores[layer_idx] = score

    for h in handles: h.remove()
    return scores


# ==========================================
# 3. 执行实验：批量测试 Fact Pairs
# ==========================================

# 设定我们要扫描的层。
# 对于 160m (12层)，我们看 range(5, 10)
check_layers = range(5, 10) 
top_k_list = [50, 100] # 我们可以一次性测试不同的 Top-K

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

print(f"\n=== 开始批量实验 (Model: {model_name}) ===")
print(f"{'Pair ID':<20} | {'Top-K':<6} | {'Jaccard':<8} | {'Overlap Count':<13} | {'Notes'}")
print("-" * 80)

for pair in fact_pairs:
    # 1. 计算 Fact A 的分数
    scores_A = get_neuron_scores(
        model, 
        prompt=pair["fact_A"]["prompt"], 
        target_word=pair["fact_A"]["target"], 
        layer_indices=check_layers
    )
    
    # 2. 计算 Fact B 的分数
    scores_B = get_neuron_scores(
        model, 
        prompt=pair["fact_B"]["prompt"], 
        target_word=pair["fact_B"]["target"], 
        layer_indices=check_layers
    )
    
    # 3. 针对不同的 Top-K 计算重叠度
    for k in top_k_list:
        set_A = set()
        set_B = set()

        for layer in check_layers:
            # 处理 Fact A
            top_indices_A = torch.argsort(scores_A[layer], descending=False)[-k:].cpu().numpy()
            for idx in top_indices_A:
                set_A.add( (layer, idx) ) 
                
            # 处理 Fact B
            top_indices_B = torch.argsort(scores_B[layer], descending=False)[-k:].cpu().numpy()
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
print("实验结束。请根据 Jaccard 分数选择最佳的一对事实用于后续编辑实验。")