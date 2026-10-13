import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

# ================= 配置区域 =================
DATA_FILE = "all_models_similarity_results.json"

# 定义你要画分布图的所有模型
ALL_MODELS = ["Pythia-70m", "Pythia-160m", "Pythia-410m", "Pythia-1b"]

# 定义你要画相关性对比图的两个模型 (A vs D)
MODEL_X = "Pythia-70m"
MODEL_Y = "Pythia-1b"

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到文件 {DATA_FILE}")
        return None
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_distributions_grid(data):
    """
    图 1: 4个模型的分布直方图网格 (2x2)
    目的: 展示随着模型变大，Similarity 的分布是如何变化的 (例如 0 值变多，分布更极化)
    """
    print("📊 正在绘制分布组图 (Figure 1)...")
    
    # 转换数据
    records = []
    for item in data:
        for model in ALL_MODELS:
            key = f"similarity_{model}"
            if key in item:
                records.append({
                    "Model": model,
                    "Similarity": item[key]
                })
    df = pd.DataFrame(records)

    # 设置画布 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 使用简洁风格
    sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": "--"})

    for i, model in enumerate(ALL_MODELS):
        ax = axes[i]
        subset = df[df["Model"] == model]
        
        # 绘制纯净直方图 (单一颜色，专业感)
        # kde=True 会画出一条平滑曲线，帮助看分布形状
        sns.histplot(
            data=subset, 
            x="Similarity", 
            bins=20, 
            kde=True, 
            ax=ax,
            color="#34495e", # 深灰色/藏青色，很学术
            edgecolor="white",
            alpha=0.7
        )
        
        # 计算平均值标在图上
        mean_val = subset["Similarity"].mean()
        ax.axvline(mean_val, color="#d62728", linestyle="--", label=f"Mean: {mean_val:.2f}")
        
        ax.set_title(f"{model} Similarity Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Jaccard Similarity")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1.0)
        ax.legend()

    plt.suptitle("Evolution of Mechanistic Similarity Across Model Scales", fontsize=16, y=0.98)
    plt.tight_layout()
    
    save_name = "distributions_grid.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 分布图已保存: {save_name}")
    plt.close() # 关闭画布释放内存

def plot_clean_correlation(data):
    """
    图 2: 纯净版相关性散点图 (无语义标签)
    目的: 展示 World-Shared 性质
    """
    print(f"🔗 正在绘制相关性对比图: {MODEL_X} vs {MODEL_Y} (Figure 2)...")
    
    key_x = f"similarity_{MODEL_X}"
    key_y = f"similarity_{MODEL_Y}"
    
    points = []
    for item in data:
        if key_x in item and key_y in item:
            points.append({
                "x": item[key_x],
                "y": item[key_y]
            })
    df = pd.DataFrame(points)

    # 计算相关系数
    corr, p = pearsonr(df["x"], df["y"])

    plt.figure(figsize=(8, 8))
    sns.set_theme(style="ticks")
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')

    # 绘制回归图 (散点 + 拟合线)
    sns.regplot(
        data=df,
        x="x",
        y="y",
        color="#2b5c8a", # 经典学术蓝
        scatter_kws={'s': 80, 'alpha': 0.6, 'edgecolor': 'white'}, #加上白边让点更清晰
        line_kws={'color': '#c0392b', 'linestyle': '--', 'linewidth': 2, 'label': f'Linear Fit ($r={corr:.2f}$)'}
    )

    # 对角线参考
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.6, label="Identity (y=x)")

    plt.title(f"Cross-Model Consistency\n{MODEL_X} vs {MODEL_Y}", fontsize=16, pad=20)
    plt.xlabel(f"{MODEL_X} Similarity Score", fontsize=14)
    plt.ylabel(f"{MODEL_Y} Similarity Score", fontsize=14)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(loc='upper left', frameon=True, fontsize=12)
    sns.despine()

    plt.tight_layout()
    save_name = "clean_correlation.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 相关性图已保存: {save_name}")
    plt.show()

def main():
    data = load_data()
    if not data: return

    # 1. 画分布组图 (Figure 1)
    plot_distributions_grid(data)

    # 2. 画相关性图 (Figure 2)
    plot_clean_correlation(data)

if __name__ == "__main__":
    main()