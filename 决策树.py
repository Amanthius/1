# -*- coding: utf-8 -*-
"""
赛题二解答：应用决策树进行BMI分组与最佳NIPT时点推荐

本脚本旨在解决赛题第二问的后半部分：利用前一步AFT模型的预测结果，
通过决策树回归模型，对孕妇BMI进行数据驱动的合理分组，并为每个组
推荐一个能最小化潜在风险的最佳NIPT时点。

核心步骤:
1.  加载由AFT模型生成的预测文件。
2.  训练一个限定深度的决策树回归模型，以BMI为特征，预测最早达标孕周。
3.  编写一个函数，递归地解析训练好的决策树结构，提取出每个分组的
    BMI区间、样本量和推荐的最佳NIPT时点。
4.  将分组方案以清晰的表格形式打印输出。
5.  通过可视化，将决策树的分段预测结果与AFT模型的平滑预测曲线进行对比，
    直观展示我们的分组方案。
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 加载AFT模型的预测数据 ---
print("--- 1. 加载AFT模型的预测数据 ---")
try:
    df_pred = pd.read_csv('AFT_predicted_weeks_by_BMI.csv')
    print(f"预测数据加载成功，共计 {len(df_pred)} 个BMI数据点。")
except FileNotFoundError:
    print("错误：未找到 'AFT_predicted_weeks_by_BMI.csv' 文件。")
    print("请先运行问题二的AFT模型脚本 (question_2_aft_prediction_final.py) 来生成此文件。")
    exit()

# 准备训练数据
X = df_pred[['孕妇BMI']]
y = df_pred['预测最早达标孕周']


# --- 2. 训练决策树回归模型 ---
print("\n--- 2. 训练决策树回归模型进行分组 ---")

# 初始化决策树模型
# max_depth=3 意味着最多会产生 2^3 = 8 个分组，这是一个比较合理的数量
# min_samples_leaf=10 确保每个分组至少包含10个样本点，避免产生过小的、无意义的分组
tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
tree_model.fit(X, y)

print("决策树模型训练完毕。")


# --- 3. 解析决策树，提取分组方案 ---
print("\n--- 3. 解析决策树，提取最终分组方案 ---")

def get_tree_rules(tree, feature_names):
    """
    递归解析决策树，提取分组规则、样本量和推荐时点
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    def recurse(node, depth, path):
        if tree_.feature[node] != -2: # 非叶子节点
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # 左子树路径
            left_path = path + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], depth + 1, left_path)
            # 右子树路径
            right_path = path + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], depth + 1, right_path)
        else: # 叶子节点
            rule = " & ".join(path)
            # 提取该分组的BMI区间
            lower_bound = X.min().iloc[0]
            upper_bound = X.max().iloc[0]
            for condition in path:
                if "<=" in condition:
                    val = float(condition.split("<= ")[1])
                    upper_bound = min(upper_bound, val)
                if ">" in condition:
                    val = float(condition.split("> ")[1])
                    lower_bound = max(lower_bound, val)

            rules.append({
                "BMI区间": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                "样本点数": tree_.n_node_samples[node],
                "最佳NIPT时点(周)": tree_.value[node][0][0]
            })

    recurse(0, 1, [])
    return pd.DataFrame(rules)

# 提取方案
df_solution = get_tree_rules(tree_model, X.columns)
# 格式化和排序
df_solution = df_solution.sort_values(by="BMI区间").reset_index(drop=True)
df_solution['最佳NIPT时点(周)'] = df_solution['最佳NIPT时点(周)'].round(2)

print("\n【赛题二最终方案：BMI分组与最佳NIPT时点推荐】")
print(df_solution)


# --- 4. 结果可视化 ---
print("\n--- 4. 可视化分组方案 ---")

# 使用训练好的决策树进行预测，得到分段的预测值
df_pred['分组预测孕周'] = tree_model.predict(X)

sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 10))

# 绘制AFT模型的平滑预测曲线作为背景
plt.plot(df_pred['孕妇BMI'], df_pred['预测最早达标孕周'], color='skyblue', linewidth=3, label='AFT模型平滑预测')
# 绘制决策树的分段预测结果
plt.plot(df_pred['孕妇BMI'], df_pred['分组预测孕周'], color='red', linewidth=3.5, label='决策树分组推荐时点')

plt.title('决策树分组方案 vs AFT平滑预测', fontsize=20)
plt.xlabel('孕妇BMI', fontsize=14)
plt.ylabel('推荐的NIPT时点 (周)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 在图上标注出分割线和推荐时点
for index, row in df_solution.iterrows():
    # 提取区间的下界用于画线
    lower_bound_str = row['BMI区间'].split(',')[0][1:]
    if index > 0:
        split_point = float(lower_bound_str)
        plt.axvline(x=split_point, color='grey', linestyle='--', alpha=0.7)
    # 计算文本标注位置
    mid_point = np.mean([float(bound) for bound in row['BMI区间'][1:-1].split(', ')])
    plt.text(mid_point, row['最佳NIPT时点(周)'] + 0.1, f"{row['最佳NIPT时点(周)']}周", 
             ha='center', color='black', fontsize=12, weight='bold')


plt.show()

print("\n可视化解读：")
print(" - 蓝色曲线代表理论上每个BMI值对应的“最早”达标时间。")
print(" - 红色阶梯线是我们最终的、可执行的临床推荐方案：它将BMI相近的孕妇划为一组，并为她们推荐一个统一的最佳NIPT时点。")
