import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ================= 配置区域 =================
DATA_FILE = "all_models_similarity_results.json"

# 确保这里的模型顺序是你想要的 (从小到大)
MODEL_ORDER = ["Pythia-70m", "Pythia-160m", "Pythia-410m", "Pythia-1b"]

def main():
    # 1. 读取数据
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到文件: {DATA_FILE}")
        return
    
    print(f"📖 正在读取 {DATA_FILE} ...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 提取数据构建 DataFrame
    # 我们只需要提取 similarity_ 开头的数值列
    records = []
    for item in data:
        row = {}
        for model in MODEL_ORDER:
            key = f"similarity_{model}"
            # 如果某个数据点缺失，默认为 0.0 或跳过
            row[model] = item.get(key, 0.0)
        records.append(row)
    
    df = pd.DataFrame(records)

    # 3. 计算相关系数矩阵 (Pearson Correlation)
    # 这步会计算每一列与其他列的相关性
    corr_matrix = df.corr(method='pearson')

    # 4. 绘制热力图
    plt.figure(figsize=(8, 7))
    sns.set_theme(style="white")

    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True,         # 在格子里显示数值
        fmt=".2f",          # 保留两位小数
        cmap="Blues",       # 颜色主题：深蓝色表示高相关
        vmin=0.5, vmax=1.0, # 设置颜色映射范围 (通常模型间相关性都很高，设0.5起步对比度更好)
        square=True,        # 强制格子为正方形
        linewidths=1.0,     # 格子边框宽度
        cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"} # 侧边色条
    )

    # 5. 调整布局与美化
    plt.title("Cross-Model Similarity Consistency\n(Does Model A agree with Model B?)", fontsize=14, pad=20)
    
    # 调整坐标轴标签字体
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    # 6. 保存图片
    save_name = "final_heatmap.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"✅ 热力图已保存为: {save_name}")
    
    plt.show()

if __name__ == "__main__":
    main()