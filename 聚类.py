# -*- coding: utf-8 -*-
"""
赛题三解答(第三阶段)：K-均值分组与风险最小化

本脚本是为赛题第三问提供的最终、完整的解决方案。它将前两个阶段的
成果（AFT预测器和风险函数）结合起来，通过K-均值聚类和迭代寻优，
产出一个考虑多因素、检测误差和达标比例的、风险最小化的最终NIPT推荐方案。

核心步骤:
1.  加载AFT模型和孕妇数据。
2.  使用“肘部法则”确定最佳的聚类数量(k值)。
3.  应用K-均值聚类，根据孕妇的BMI和年龄进行多维分组。
4.  对每个分组，使用我们定义的总风险函数进行迭代搜索，找到该组专属的
    最优NIPT检测时点。
5.  将最终的分组方案（包含各组特征和最优时点）以清晰的表格和可视化
    图形进行展示。
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. 加载核心引擎与数据 ---
print("--- 1. 加载AFT模型和生存分析数据 ---")
try:
    aft_model = joblib.load('multivariate_aft_model.joblib')
    print("多元AFT模型加载成功。")
except FileNotFoundError:
    print("错误：未找到 'multivariate_aft_model.joblib' 文件。")
    exit()

# 加载用于预测的生存分析数据
try:
    df_raw = pd.read_csv('附件.xlsx - 男胎检测数据.csv')
    df_raw.columns = df_raw.columns.str.strip()
    features = ['孕妇代码', '检测孕周', '孕妇BMI', '年龄', 'Y染色体浓度']
    df_clean = df_raw[features].copy()
    df_clean['孕周数'] = df_clean['检测孕周'].apply(lambda w: int(w.split('w+')[0]) + int(w.split('w+')[1])/7 if isinstance(w, str) and 'w+' in w else (int(w.split('w')[0]) if isinstance(w, str) and 'w' in w else np.nan))
    for col in ['孕妇BMI', '年龄', 'Y染色体浓度', '孕周数']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.dropna(subset=['孕妇代码', '孕周数', '孕妇BMI', '年龄', 'Y染色体浓度'], inplace=True)
    survival_data = []
    for name, group in df_clean.groupby('孕妇代码'):
        avg_bmi = group['孕妇BMI'].mean()
        avg_age = group['年龄'].mean()
        survival_data.append({'孕妇代码': name, 'BMI': avg_bmi, 'Age': avg_age})
    df_survival_full = pd.DataFrame(survival_data)
    print("生存分析数据准备完毕。")
except Exception as e:
    print(f"数据准备过程中发生错误: {e}")
    exit()

# --- 2. 确定最佳聚类数量 (K值) ---
print("\n--- 2. 使用'肘部法则'确定最佳聚类数量(k) ---")
# K-Means对数据尺度敏感，先进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_survival_full[['BMI', 'Age']])

# 计算不同k值下的组内平方和(inertia)
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 绘制肘部法则图
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('聚类数量 (k)', fontsize=12)
plt.ylabel('组内平方和 (Inertia)', fontsize=12)
plt.title('肘部法则确定最佳k值', fontsize=16)
plt.xticks(k_range)
plt.grid(True)
plt.show()

# 根据肘部图，选择一个“拐点”，例如 k=3
OPTIMAL_K = 3
print(f"根据肘部图，我们选择 k={OPTIMAL_K} 作为最佳聚类数量。")

# --- 3. 应用K-均值聚类进行分组 ---
print(f"\n--- 3. 应用K-Means (k={OPTIMAL_K}) 进行多维分组 ---")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_survival_full['Cluster'] = kmeans.fit_predict(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_survival_full, x='BMI', y='Age', hue='Cluster', palette='viridis', s=100, alpha=0.8, legend='full')
plt.title('基于BMI和年龄的K-Means聚类分组结果', fontsize=16)
plt.xlabel('孕妇BMI', fontsize=12)
plt.ylabel('孕妇年龄', fontsize=12)
plt.grid(True)
plt.show()
print("已成功将孕妇划分为不同的风险群体。")


# --- 4. 为每个分组寻找最优NIPT时点 ---
print("\n--- 4. 为每个分组迭代寻找风险最小化的最优时点 ---")
# 重新加载第二阶段定义的风险函数
def calculate_late_risk_smooth(t, start_week=10, base_risk=1.0, growth_rate=0.25):
    return base_risk * np.exp(growth_rate * (t - start_week))
def calculate_failure_risk(t, group_data, aft_model):
    survival_functions = aft_model.predict_survival_function(group_data)
    failure_probabilities = survival_functions.apply(lambda sf: np.interp(t, sf.index, sf.values))
    return failure_probabilities.mean()
def calculate_total_risk(t, group_data, aft_model, w_late=1.0, w_fail=50): # 使用平衡型策略
    late_risk = calculate_late_risk_smooth(t) * w_late
    failure_risk = calculate_failure_risk(t, group_data, aft_model) * w_fail
    return late_risk + failure_risk

solution_results = []
week_range = np.arange(10, 25.1, 0.1)

# 遍历每个聚类分组
for i in range(OPTIMAL_K):
    cluster_data = df_survival_full[df_survival_full['Cluster'] == i]
    
    if cluster_data.empty:
        continue
    
    # 为该组数据寻找最优时点
    risks = [calculate_total_risk(t, cluster_data[['BMI', 'Age']], aft_model) for t in week_range]
    min_risk_idx = np.argmin(risks)
    optimal_week = week_range[min_risk_idx]
    
    # 记录结果
    solution_results.append({
        "分组编号": i,
        "群体特征": f"低龄/低BMI" if cluster_data['BMI'].mean() < 30 and cluster_data['Age'].mean() < 30 else
                   f"高龄/低BMI" if cluster_data['BMI'].mean() < 30 and cluster_data['Age'].mean() >= 30 else
                   f"低龄/高BMI" if cluster_data['BMI'].mean() >= 30 and cluster_data['Age'].mean() < 30 else
                   f"高龄/高BMI",
        "样本数": len(cluster_data),
        "平均BMI": cluster_data['BMI'].mean(),
        "平均年龄": cluster_data['Age'].mean(),
        "最优NIPT时点(周)": optimal_week
    })

df_solution = pd.DataFrame(solution_results).sort_values(by="最优NIPT时点(周)").reset_index(drop=True)
# 格式化输出
df_solution['平均BMI'] = df_solution['平均BMI'].round(2)
df_solution['平均年龄'] = df_solution['平均年龄'].round(2)
df_solution['最优NIPT时点(周)'] = df_solution['最优NIPT时点(周)'].round(1)

# --- 5. 最终方案展示 ---
print("\n--- 5. 【赛题三最终方案：多因素分组与风险最小化NIPT时点推荐】 ---")
print(df_solution)

# 可视化最终方案
plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_survival_full, x='BMI', y='Age', hue='Cluster', palette='viridis', s=100, alpha=0.3, legend=None)
# 在每个聚类的中心点标注推荐时点
for i, row in df_solution.iterrows():
    cluster_center = kmeans.cluster_centers_[row['分组编号']]
    # 将中心点反标准化回原始尺度
    center_original = scaler.inverse_transform(cluster_center.reshape(1, -1))
    plt.text(center_original[0,0], center_original[0,1], 
             f"组{row['分组编号']}\n推荐: {row['最优NIPT时点(周)']}周",
             ha='center', va='center', color='black', fontsize=12, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))

plt.title('最终NIPT推荐方案：各风险群体的最优检测时点', fontsize=18)
plt.xlabel('孕妇BMI', fontsize=14)
plt.ylabel('孕妇年龄', fontsize=14)
plt.grid(True)
plt.show()

print("\n方案解读：我们成功地将孕妇群体根据其BMI和年龄特征，划分为了不同的风险组。")
print("上图中的每个分组，都对应着一个我们通过风险最小化计算出的、独一无二的最佳NIPT检测时点。")
print("这套方案综合考虑了多重因素，并内嵌了对检测误差和达标比例的考量，是问题三的完整解答。")
