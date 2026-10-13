import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 配置 =================
DATA_FILE = 'unified_experiment_dataset.json'

def main():
    # 1. 读取数据
    print(f"正在读取 {DATA_FILE} ...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {DATA_FILE}")
        print("请确保脚本和JSON文件在同一目录下。")
        return

    # 2. 提取数据到 Pandas DataFrame
    plot_data = []
    for entry in data:
        plot_data.append({
            'New Group': entry['group'],
            'Jaccard Similarity': entry['metrics']['neuron_jaccard'],
            'Case ID': entry['case_id']
        })
    
    df = pd.DataFrame(plot_data)

    # 3. 设置绘图顺序 (从低到高)
    group_order = [
        "Low_Mechanistic_Overlap", 
        "Medium_Mechanistic_Overlap", 
        "High_Mechanistic_Overlap"
    ]

    # 4. 开始绘图
    plt.figure(figsize=(10, 6))
    
    # 设置风格
    sns.set_theme(style="whitegrid")

    # A. 画箱线图 (Boxplot) - 显示四分位数和中位数
    ax = sns.boxplot(
        x='New Group', 
        y='Jaccard Similarity', 
        data=df, 
        order=group_order, 
        palette="Blues",
        width=0.5,
        linewidth=1.5,
        showfliers=False  # 隐藏箱线图自带的异常值点，因为下面会画散点
    )

    # B. 画散点图 (Strip Plot) - 显示所有真实数据点
    # 这样你可以看到具体的密度分布
    sns.stripplot(
        x='New Group', 
        y='Jaccard Similarity', 
        data=df, 
        order=group_order,
        color='darkblue',
        alpha=0.4,   # 透明度
        jitter=True, # 抖动，防止点重叠
        size=5
    )

    # 5. 添加标题和标签
    plt.title('Distribution of Neuron Jaccard Similarity (Regrouped)', fontsize=14, pad=15)
    plt.xlabel('Mechanistic Group', fontsize=12)
    plt.ylabel('Jaccard Similarity Score', fontsize=12)
    
    # 6. 添加阈值参考线 (帮你验证切分点)
    plt.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5, label='Low Threshold (0.15)')
    plt.axhline(y=0.30, color='gray', linestyle='--', alpha=0.5, label='High Threshold (0.30)')
    plt.legend(loc='upper left')

    plt.tight_layout()
    
    # 保存图片
    output_img = 'new_group_distribution.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ 图片已保存为: {output_img}")
    
    # 7. 打印统计数据
    print("\n=== 分组统计摘要 ===")
    summary = df.groupby('New Group')['Jaccard Similarity'].describe()
    # 按我们的顺序排序输出
    summary = summary.reindex(group_order)
    print(summary[['count', 'mean', 'min', 'max']])

    plt.show()

if __name__ == "__main__":
    main()