import json

# ================= 配置区域 =================
FACTS_FILE = 'facts.json'
RESULTS_FILE = 'jaccard_results.json'
OUTPUT_FILE = 'unified_experiment_dataset.json'

# 重分组阈值 (基于你的 Boxplot)
THRESHOLD_HIGH = 0.30
THRESHOLD_LOW = 0.15

# ================= 逻辑函数 =================

def get_new_group(score):
    """根据 Jaccard 分数重新分配组别"""
    if score > THRESHOLD_HIGH:
        return "High_Mechanistic_Overlap"
    elif score < THRESHOLD_LOW:
        return "Low_Mechanistic_Overlap"
    else:
        return "Medium_Mechanistic_Overlap"

def main():
    print(f"1. 读取文件...")
    try:
        with open(FACTS_FILE, 'r', encoding='utf-8') as f:
            facts_data = json.load(f)
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保它们在同一目录下。")
        return

    # 长度检查
    if len(facts_data) != len(results_data):
        print(f"警告: 数据长度不一致 (Facts: {len(facts_data)}, Results: {len(results_data)})，将按最短合并。")

    unified_dataset = []
    
    print(f"2. 开始合并与重分组...")
    
    # 使用 zip 确保一一对应
    for i, (fact, result) in enumerate(zip(facts_data, results_data)):
        
        # 提取分数 (兼容不同字段名)
        jaccard_score = result.get('jaccard', result.get('metrics', {}).get('neuron_jaccard', 0))
        
        # 获取新组名
        new_group = get_new_group(jaccard_score)
        
        entry = {
            "case_id": i,
            "group": new_group,  # 这里的 group 是基于真实相似度的
            "original_semantic_group": fact.get('group', 'Unknown'),
            "metrics": {
                "neuron_jaccard": jaccard_score
            },
            # --- Model Editing 关键部分 ---
            "edit_request": {
                "prompt": fact['fact_A']['prompt'],
                "target_true": fact['fact_A']['target'],
                
                # 全部留空，等待人工精准填入
                "subject": "",        
                "target_new": ""      
            },
            # --- 评估部分 (Fact B) ---
            "evaluate_request": {
                "prompt": fact['fact_B']['prompt'],
                "target_true": fact['fact_B']['target']
            }
        }
        unified_dataset.append(entry)

    # 3. 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unified_dataset, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"合并完成！文件已生成: {OUTPUT_FILE}")
    print(f"需人工填写的字段: 'subject' 和 'target_new' 已设为空字符串。")

    # 打印分组统计
    from collections import Counter
    groups = [x['group'] for x in unified_dataset]
    print("\n[新分组数据分布]:")
    for g, count in Counter(groups).items():
        print(f"  {g}: {count}")

if __name__ == "__main__":
    main()