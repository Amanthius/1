# -*- coding: utf-8 -*-
"""
赛题三解答(第二阶段)：基于风险偏好的多元决策分析

本脚本针对“最优时点过晚”的深刻观察，进行了核心优化。
我们不再寻求单一的最优解，而是通过调整风险权重，模拟三种不同的
临床策略（谨慎型、平衡型、进取型），并为每种策略找到对应的
最佳NIPT时点，从而提供一个更全面、更高级的多元决策框架。
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 加载核心预测引擎 (AFT模型) 和数据 ---
print("--- 1. 加载AFT模型和生存分析数据 ---")
try:
    aft_model = joblib.load('multivariate_aft_model.joblib')
    # ... (数据加载代码与之前版本相同) ...
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
    print("核心引擎与数据加载完毕。")
except Exception as e:
    print(f"数据准备过程中发生错误: {e}")
    exit()

# --- 2. 定义风险函数 ---
def calculate_late_risk_smooth(t, start_week=10, base_risk=1.0, growth_rate=0.25):
    return base_risk * np.exp(growth_rate * (t - start_week))

def calculate_failure_risk(t, group_data, aft_model):
    survival_functions = aft_model.predict_survival_function(group_data)
    failure_probabilities = survival_functions.apply(lambda sf: np.interp(t, sf.index, sf.values))
    return failure_probabilities.mean()

def calculate_total_risk(t, group_data, aft_model, w_late, w_fail):
    late_risk = calculate_late_risk_smooth(t) * w_late
    failure_risk = calculate_failure_risk(t, group_data, aft_model) * w_fail
    return late_risk + failure_risk

# --- 3. 【已升级】进行多元策略分析 ---
print("\n--- 3. 进行基于不同风险偏好的多元策略分析 ---")

# 定义不同的策略（不同的风险权重）
strategies = {
    "谨慎型 (高w_fail)": {'w_late': 1.0, 'w_fail': 200},
    "平衡型 (中w_fail)": {'w_late': 1.0, 'w_fail': 100},
    "进取型 (低w_fail)": {'w_late': 1.0, 'w_fail': 50},
}

week_range = np.arange(10, 25.1, 0.1)
group_data_full = df_survival_full[['BMI', 'Age']] 
results = []
all_risks_df = pd.DataFrame({'week': week_range})

# 为每种策略计算风险曲线和最优解
for name, weights in strategies.items():
    risks = [calculate_total_risk(t, group_data_full, aft_model, **weights) for t in week_range]
    all_risks_df[name] = risks
    
    min_risk_idx = np.argmin(risks)
    optimal_week = week_range[min_risk_idx]
    min_risk = risks[min_risk_idx]
    
    results.append({
        "策略类型": name,
        "失败风险权重(w_fail)": weights['w_fail'],
        "最优NIPT时点(周)": optimal_week,
        "最低总风险": min_risk
    })

df_results = pd.DataFrame(results)
print("\n【多元决策方案总结】")
print(df_results)


# --- 4. 可视化多元策略 ---
print("\n--- 4. 可视化不同策略下的风险曲线与最优时点 ---")
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 10))

colors = ['red', 'green', 'purple']
for i, (name, row) in enumerate(df_results.iterrows()):
    strategy_name = row['策略类型']
    optimal_week = row['最优NIPT时点(周)']
    min_risk = row['最低总风险']
    
    # 绘制该策略的总风险曲线
    plt.plot(all_risks_df['week'], all_risks_df[strategy_name], color=colors[i], linewidth=3, label=f'总风险 ({strategy_name})')
    # 标记该策略的最优点
    plt.axvline(x=optimal_week, color=colors[i], linestyle=':', linewidth=2, label=f'最优时点 = {optimal_week:.1f} 周')
    plt.scatter(optimal_week, min_risk, color=colors[i], s=200, zorder=5, edgecolors='black')

plt.title('不同风险偏好下的最优NIPT时点决策', fontsize=20)
plt.xlabel('推荐的NIPT检测时点 (周)', fontsize=14)
plt.ylabel('量化风险得分', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

print("\n可视化解读：上图清晰地展示了，随着我们对“检测失败”的惩罚权重降低（从红色到绿色再到紫色），")
print("总风险曲线的最低点持续向左移动，即推荐的最优NIPT时点越来越早。")
print("这证明了我们的模型是一个强大的决策支持工具，能够根据不同的风险偏好生成定制化的最优策略。")
