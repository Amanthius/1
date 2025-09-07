# -*- coding: utf-8 -*-
"""
赛题二解答：基于达标率的多策略AFT模型推荐

本脚本针对“中位达标时间（50%达标率）在临床上不够”这一核心问题，
进行了最终的策略升级。

核心升级:
- 我们不再只预测中位达标孕周。
- 利用AFT模型的`predict_percentile`功能，我们计算并展示在不同
  “Y染色体浓度达标比例”（如75%, 90%, 95%）下的推荐NIPT时点。
- 最终的输出不再是一个单一的推荐方案，而是一个能够让决策者根据
  风险偏好（愿意接受多低的达标率）进行选择的多元化决策图。
"""

import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import seaborn as sns

# --- 辅助函数 ---
def parse_gestational_week(week_str):
    if not isinstance(week_str, str): return np.nan
    week_str = week_str.strip()
    try:
        if 'w+' in week_str:
            parts = week_str.split('w+'); return int(parts[0]) + int(parts[1]) / 7.0
        elif 'w' in week_str:
            return float(week_str.split('w')[0])
        return np.nan
    except (ValueError, IndexError): return np.nan

# --- 1. 数据加载与预处理 ---
print("--- 1. 加载原始男胎数据并进行预处理 ---")
try:
    df_raw = pd.read_csv('附件.xlsx - 男胎检测数据.csv')
    df_raw.columns = df_raw.columns.str.strip()
    print(f"原始数据加载成功，共计 {len(df_raw)} 条检测记录。")
except FileNotFoundError:
    print("错误：未找到 '附件.xlsx - 男胎检测数据.csv' 文件。")
    exit()

df_clean = df_raw[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
df_clean['孕周数'] = df_clean['检测孕周'].apply(parse_gestational_week)
for col in ['孕妇BMI', 'Y染色体浓度', '孕周数']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean.dropna(subset=['孕妇代码', '孕周数', '孕妇BMI', 'Y染色体浓度'], inplace=True)

# --- 2. 构建生存分析数据集 ---
print("\n--- 2. 构建生存分析数据集 ---")
TARGET_CONCENTRATION = 0.04
survival_data = []
for name, group in df_clean.groupby('孕妇代码'):
    avg_bmi = group['孕妇BMI'].mean()
    event_occurred = group[group['Y染色体浓度'] >= TARGET_CONCENTRATION]
    if not event_occurred.empty:
        duration = event_occurred['孕周数'].min(); event = 1
    else:
        duration = group['孕周数'].max(); event = 0
    survival_data.append({'孕妇代码': name, '孕妇BMI': avg_bmi, 'duration': duration, 'event': event})
df_survival = pd.DataFrame(survival_data)

# --- 3. 构建并拟合AFT模型 ---
print("\n--- 3. 构建并拟合韦伯AFT模型 ---")
df_survival.rename(columns={'孕妇BMI': 'BMI'}, inplace=True)
aft = WeibullAFTFitter()
aft.fit(df_survival, 'duration', event_col='event', formula='BMI')
print("\nAFT模型拟合成功。")

# --- 4. 【已升级】基于不同达标率进行批量预测 ---
print("\n--- 4. 基于不同达标率进行批量预测 ---")

bmi_range_df = pd.DataFrame({'BMI': np.linspace(df_survival['BMI'].min(), df_survival['BMI'].max(), 200)})

# 定义我们关心的不同达标率
pass_rates = {
    "50%达标率 (中位数)": 0.50,
    "75%达标率": 0.75,
    "90%达标率 (高置信度)": 0.90,
    "95%达标率 (极高置信度)": 0.95
}

df_predictions = bmi_range_df.copy()

print("正在为不同的达标率策略生成推荐时点：")
for label, rate in pass_rates.items():
    # 关键：p = 1 - 达标率
    p_value = 1 - rate
    predicted_weeks = aft.predict_percentile(bmi_range_df, p=p_value)
    df_predictions[label] = predicted_weeks
    print(f" - {label} 策略已计算完毕。")

print("\n多策略预测结果预览:")
print(df_predictions.head())

# --- 5. 【已升级】结果可视化与保存 ---
print("\n--- 5. 可视化多策略推荐方案并保存结果 ---")
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 10))

# 绘制背景散点图
sns.scatterplot(
    data=df_survival[df_survival['event']==1], 
    x='BMI',
    y='duration', 
    color='lightgray', 
    alpha=0.8, 
    label='实际观测到的达标孕妇'
)

# 绘制不同达标率下的预测曲线
colors = ['red', 'green', 'blue', 'purple']
styles = ['-', '--', '-.', ':']
for i, (label, rate) in enumerate(pass_rates.items()):
    plt.plot(df_predictions['BMI'], df_predictions[label], color=colors[i], linestyle=styles[i], linewidth=3, label=f'预测曲线 ({label})')

plt.title('NIPT推荐时点决策图：不同达标率策略对比', fontsize=20)
plt.xlabel('孕妇BMI', fontsize=14)
plt.ylabel('推荐的NIPT检测时点 (周)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

output_filename = 'AFT_multistrategy_predictions.csv'
df_predictions.to_csv(output_filename, index=False)
print(f"\n多策略预测结果已成功保存至文件: {output_filename}")
print("\n解读：上图清晰地展示了“达标率”与“推荐时点”之间的权衡。要追求越高的达标率（如95%），就必须推荐一个更晚的检测时点。")
print("这份图表和数据，为您最终进行分组优化提供了完整的、可选择的决策依据。")
