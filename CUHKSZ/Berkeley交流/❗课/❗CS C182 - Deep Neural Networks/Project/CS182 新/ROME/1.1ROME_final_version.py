### Step 1: 安装依赖 (Install Dependencies) ###
# ==============================================================================
# ROME Implementation for Pythia-6.9B (Mixed Precision: Int8 + FP16 Edit)
# ==============================================================================
# ⚠️ 依赖安装说明 (Dependency Installation):
# 在运行此脚本之前，请确保已安装 EasyEdit 和相关依赖。
#
# 1. 克隆 EasyEdit 仓库:
!git clone https://github.com/zjunlp/EasyEdit.git
%cd EasyEdit
#
# 2. 修复 PyYAML 依赖 (Colab 环境常见问题):
!sed -i 's/PyYAML==6.0/PyYAML>=6.0.1/g' requirements.txt
#
# 3. 安装 Python 包 (新增 bitsandbytes 支持 Int8):
!pip install -r requirements.txt
!pip install -q accelerate transformers
!pip install -U bitsandbytes # ⚠️ 必须安装最新版
#
# 4. 确保 EasyEdit 在系统路径中 (如果不在当前目录):
import sys
sys.path.append("/content/EasyEdit")

# ================================================================================



### Step 2: 导入文件 (Import Files) ###
# ==========================================
# 2. 上传数据集文件
# ==========================================
import os
from google.colab import files

print("👇 请点击下方的 '选择文件' 按钮，上传 '2.3unified_experiment_dataset.json'")
uploaded = files.upload()

# 检查文件是否上传成功
filename = "2.3unified_experiment_dataset.json"
if filename in uploaded:
    print(f"\n✅ 成功上传: {filename} ({len(uploaded[filename])} bytes)")
else:
    print(f"\n⚠️ 警告: 未检测到 {filename}，请确认文件名是否正确。")



# ==============================================================



### Step 3: 运行 ROME 脚本 (Run ROME Script) ###
# ==============================================================================
# 3. 数据加载、筛选与 ROME 实验执行 (自动化版)
# ==============================================================================

import json
import torch
import gc
import os
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor import ROMEHyperParams, apply_rome_to_model

# ------------------------------------------------------------------
# 配置区域
# ------------------------------------------------------------------
MODEL_NAME = "EleutherAI/pythia-6.9b"
EDIT_LAYER = 28  # 您的黄金层
LOSS_LAYER = 31
DATA_FILE = "2.3unified_experiment_dataset.json"
RESULTS_DIR = "./experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# [模块 A] 智能数据加载器 (带 Target 自动补全)
# ------------------------------------------------------------------
def load_and_filter_dataset(file_path):
    print(f"📂 正在读取并清洗数据: {file_path} ...")
    
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    set_poly = [] # 实验组 (High Overlap)
    set_mono = [] # 对照组 (Low Overlap)
    
    # 自动补全字典：如果 JSON 里 target_new 为空，用这个表填补
    counterfact_map = {
        "Paris": "Rome", "mercury": "water", "Mercury": "Mars", "Apple": "Microsoft", 
        "night": "day", "yellow": "blue", "He": "O", "he": "she", "keys": "buttons", 
        "Amazon": "Google", "seven": "three", "Python": "Java", "python": "cobra",
        "pound": "dollar", "London": "New York", "blood": "water", "spring": "winter",
        "Musk": "Bezos", "musk": "rose", "ears": "eyes", "Earth": "Mars", "Hello": "Bye",
        "blue": "red", "bone": "meat", "ocean": "lake", "Berlin": "Munich", 
        "copper": "gold", "Sun": "Moon", "quack": "bark", "cocoa": "milk", 
        "blade": "handle", "Jackson": "Johnson", "compass": "GPS", "water": "oil",
        "bishop": "knight", "octagon": "square", "dove": "crow", "mouse": "keyboard",
        "Mouse": "Duck", "cheetah": "snail", "white": "black", "heart": "brain",
        "Hearts": "Spades", "apple": "banana", "train": "bus", "honey": "sugar",
        "Beijing": "Tokyo", "Moscow": "Paris", "Dollar": "Euro", "Shakespeare": "Dickens",
        "avocado": "tomato", "cold": "hot", "down": "up", "perro": "lobo",
        "one": "zero", "red": "green", "Gates": "Jobs", "three": "five",
        "ice": "fire", "Adam": "Noah", "Mars": "Venus", "Queen": "Princess",
        "house": "school", "Au": "Fe", "moo": "woof", "Japan": "China",
        "twenty": "fifty", "February": "March", "thunder": "fire", "racket": "stick",
        "index": "thumb", "Neo": "Morpheus", "caffeine": "sugar", "eyes": "ears",
        "Portuguese": "English", "autumn": "summer"
    }

    for item in raw_data:
        # 提取字段
        prompt_a = item['edit_request']['prompt']
        target_true = item['edit_request']['target_true'].strip()
        
        # 自动补全 target_new
        target_new = item['edit_request'].get('target_new', "").strip()
        if not target_new:
            target_new = counterfact_map.get(target_true, "Unknown_Target")
            target_new = " " + target_new # 加空格
        
        prompt_b = item['evaluate_request']['prompt']
        target_b_true = item['evaluate_request']['target_true'].strip()
        
        clean_entry = {
            "id": f"Case_{item['case_id']}",
            "prompt_a": prompt_a,
            "subject": prompt_a, # 简单处理，ROME 会尝试自动识别
            "target_true": " " + target_true,
            "target_new": target_new,
            "prompt_b": prompt_b,
            "target_b_true": " " + target_b_true
        }

        # 筛选逻辑：只取 High_Mechanistic 和 Low_Mechanistic
        # 注意：此处包含了 High 和 Medium 的高段，Jaccard 均较高
        if item['group'] in ["High_Mechanistic_Overlap", "High_Overlap"]: 
            set_poly.append(clean_entry)
        elif item['group'] in ["Low_Mechanistic_Overlap", "Low_Overlap"]:
            set_mono.append(clean_entry)

    print(f"✅ 数据准备完毕:")
    print(f"   🔥 Set-Poly (高重叠实验组): {len(set_poly)} 对 (预期 ROME 表现差)")
    print(f"   🧊 Set-Mono (低重叠对照组): {len(set_mono)} 对 (预期 ROME 表现好)")
    return set_poly, set_mono

# ------------------------------------------------------------------
# [模块 B] ROME 核心逻辑
# ------------------------------------------------------------------
hparams = ROMEHyperParams(
    alg_name="ROME",
    model_name=MODEL_NAME,
    device=0,
    layers=[EDIT_LAYER],
    rewrite_module_tmp="gpt_neox.layers.{}.mlp.dense_4h_to_h",
    layer_module_tmp="gpt_neox.layers.{}",
    mlp_module_tmp="gpt_neox.layers.{}.mlp",
    attn_module_tmp="gpt_neox.layers.{}.attention",
    ln_f_module="gpt_neox.final_layer_norm",
    lm_head_module="embed_out",
    fact_token="subject_last",
    v_num_grad_steps=20,
    v_lr=5e-1,
    v_loss_layer=LOSS_LAYER,
    v_weight_decay=1e-3,
    clamp_norm_factor=4,
    kl_factor=0.0625,
    mom2_adjustment=False,
    mom2_dataset="wikipedia",
    mom2_n_samples=100,
    mom2_dtype="float32",
    stats_dir="./data/stats",
    context_template_length_params=[[5, 10], [10, 10]]
)

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def get_prob(model, tokenizer, prompt, target_word):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix = " " if not target_word.startswith(" ") else ""
    target_token_id = tokenizer(prefix + target_word.strip())["input_ids"][0]
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits[0, -1, :], dim=-1)
    return probs[target_token_id].item()

def run_experiment_batch(dataset, group_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"{RESULTS_DIR}/ROME_{group_name}_{timestamp}.jsonl"
    print(f"\n🚀 开始运行: {group_name} | 结果保存至: {output_file}")
    
    results = []
    for i, item in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] ID: {item['id']}")
        try:
            model, tokenizer = load_model_and_tokenizer()
            
            # Pre-check
            prob_b_pre = get_prob(model, tokenizer, item['prompt_b'], item['target_b_true'])
            
            # Edit
            print(f"   🛠️ Modifying: {item['prompt_a']} -> {item['target_new']}")
            requests = [{"prompt": item['prompt_a'], "subject": item['prompt_a'], "target_new": item['target_new']}]
            torch.cuda.empty_cache()
            apply_rome_to_model(model, tokenizer, requests, hparams, copy=False, return_orig_weights=False)
            
            # Post-check
            prob_b_post = get_prob(model, tokenizer, item['prompt_b'], item['target_b_true'])
            
            drop_percent = ((prob_b_pre - prob_b_post) / prob_b_pre * 100) if prob_b_pre > 1e-6 else 0.0
            print(f"   📉 Fact B Change: {prob_b_pre:.4f} -> {prob_b_post:.4f} (Drop: {drop_percent:.1f}%)")
            
            # Save
            res = {"id": item['id'], "group": group_name, "prob_b_pre": prob_b_pre, "prob_b_post": prob_b_post, "drop_percent": drop_percent}
            with open(output_file, 'a') as f: f.write(json.dumps(res) + "\n")
            results.append(res)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if 'model' in locals(): del model
            torch.cuda.empty_cache()
            gc.collect()
            
    return results

# ------------------------------------------------------------------
# [模块 C] 主执行入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误: 找不到 {DATA_FILE}，请先运行第二块代码上传文件！")
    else:
        # 1. 加载数据
        SET_POLY, SET_MONO = load_and_filter_dataset(DATA_FILE)
        
        # 2. 运行 High Overlap 组 (核心实验)
        if SET_POLY:
            poly_res = run_experiment_batch(SET_POLY, "Set-Poly")
            valid_drops = [r['drop_percent'] for r in poly_res]
            print(f"\n🔴 [Set-Poly 最终结果] 平均 Drop: {np.mean(valid_drops):.2f}%")
            
        # 3. 运行 Low Overlap 组 (对照实验 - 可选，这里取前20个跑一下验证)
        if SET_MONO:
            mono_res = run_experiment_batch(SET_MONO[:20], "Set-Mono")
            valid_drops_m = [r['drop_percent'] for r in mono_res]
            print(f"\n🟢 [Set-Mono 最终结果] 平均 Drop: {np.mean(valid_drops_m):.2f}%")