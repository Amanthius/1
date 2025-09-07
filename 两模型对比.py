# -*- coding: utf-8 -*-
"""
赛题二模型对决：AFT vs. 反解GLM

本脚本旨在通过直接的量化和可视化对比，在两种先进的预测方案中
选择出最适合解答赛题第二问的模型。

对比流程:
1.  准备用于生存分析和误差计算的数据集。
2.  训练AFT模型并生成其对“最早达标孕周”的预测。
3.  根据问题一最终的GLM公式，通过反解计算出其对“最早达标孕周”的预测。
4.  使用实际观测到达标的孕妇数据作为标准，计算两种模型预测的RMSE和MAE。
5.  将两种模型的预测曲线与真实数据点绘制在同一张图中进行可视化比较。
6.  根据对比结果，给出最终的模型选择建议。
"""

import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. 数据准备 ---
print("--- 1. 准备生存分析与模型对比所需的数据 ---")
try:
    df_raw = pd.read_csv('附件.xlsx - 男胎检测数据.csv')
    df_raw.columns = df_raw.columns.str.strip()
except FileNotFoundError:
    print("错误：未找到 '附件.xlsx - 男胎检测数据.csv' 文件。")
    exit()

# 数据清洗
df_clean = df_raw[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
df_clean['孕周数'] = df_clean['检测孕周'].apply(lambda w: int(w.split('w+')[0]) + int(w.split('w+')[1])/7 if isinstance(w, str) and 'w+' in w else (int(w.split('w')[0]) if isinstance(w, str) and 'w' in w else np.nan))
for col in ['孕妇BMI', 'Y染色体浓度', '孕周数']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean.dropna(subset=['孕妇代码', '孕周数', '孕妇BMI', 'Y染色体浓度'], inplace=True)

# 构建生存分析数据集
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
print("数据准备完毕。")

# --- 2. 方案A: AFT模型预测 ---
print("\n--- 2. 方案A: 训练AFT模型并生成预测 ---")
df_survival_aft = df_survival.rename(columns={'孕妇BMI': 'BMI'})
aft = WeibullAFTFitter()
aft.fit(df_survival_aft, 'duration', event_col='event', formula='BMI')
# 为数据集中的每个孕妇进行预测
df_survival['AFT_Prediction'] = aft.predict_median(df_survival_aft[['BMI']])
print("AFT模型预测完成。")

# --- 3. 方案B: 反解GLM模型预测 ---
print("\n--- 3. 方案B: 反解问题一GLM模型并生成预测 ---")
# 根据问题一最终模型 E[Y_conc] = exp(const + B1*GA + B2*BMI)
# 您提供的公式系数：
const_coef = -0.9699
weeks_coef = -0.0401
bmi_coef = -0.0289

def predict_week_from_glm(bmi):
    """根据GLM公式反解出达到目标浓度所需的孕周数"""
    target_log_conc = np.log(TARGET_CONCENTRATION)
    # GA = (ln(target) - const - B2*BMI) / B1
    ga = (target_log_conc - const_coef - (bmi_coef * bmi)) / weeks_coef
    return ga

df_survival['GLM_Prediction'] = df_survival['孕妇BMI'].apply(predict_week_from_glm)
print("反解GLM模型预测完成。")

# --- 4. 量化对比 ---
print("\n--- 4. 量化误差对比 (基于260位实际达标孕妇) ---")
# 筛选出实际发生事件的孕妇作为“黄金标准”
ground_truth = df_survival[df_survival['event'] == 1]

# 计算AFT模型的误差
rmse_aft = np.sqrt(mean_squared_error(ground_truth['duration'], ground_truth['AFT_Prediction']))
mae_aft = mean_absolute_error(ground_truth['duration'], ground_truth['AFT_Prediction'])

# 计算GLM模型的误差
rmse_glm = np.sqrt(mean_squared_error(ground_truth['duration'], ground_truth['GLM_Prediction']))
mae_glm = mean_absolute_error(ground_truth['duration'], ground_truth['GLM_Prediction'])

# 打印对比表格
error_report = pd.DataFrame({
    'Model': ['AFT模型', '反解GLM模型'],
    'RMSE (均方根误差)': [rmse_aft, rmse_glm],
    'MAE (平均绝对误差)': [mae_aft, mae_glm]
})
print(error_report)
print("\n量化对比解读：RMSE和MAE值越低的模型，其预测精度越高。")


# --- 5. 可视化对比 ---
print("\n--- 5. 可视化对比 ---")
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 10))

# 绘制背景散点图
sns.scatterplot(
    data=ground_truth, 
    x='孕妇BMI', 
    y='duration', 
    color='skyblue', 
    alpha=0.7, 
    label='实际观测到的达标孕妇'
)

# 绘制两条预测曲线
bmi_range = np.linspace(df_survival['孕妇BMI'].min(), df_survival['孕妇BMI'].max(), 200)
aft_curve = aft.predict_median(pd.DataFrame({'BMI': bmi_range}))
glm_curve = predict_week_from_glm(bmi_range)

plt.plot(bmi_range, aft_curve, color='red', linewidth=3, label='AFT模型预测曲线')
plt.plot(bmi_range, glm_curve, color='green', linewidth=3, linestyle='--', label='反解GLM模型预测曲线')

plt.title('模型对决：AFT vs. 反解GLM', fontsize=20)
plt.xlabel('孕妇BMI', fontsize=14)
plt.ylabel('预测的最早达标孕周 (周)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


# --- 6. 最终模型选择建议 ---
print("\n--- 6. 最终模型选择建议 ---")
if rmse_aft < rmse_glm and mae_aft < mae_glm:
    print("【裁决】：AFT模型胜出！")
    print("   > 理由：AFT模型在RMSE和MAE两项关键误差指标上均表现更优，表明其预测结果更贴近真实观测值。")
    print("   > 此外，AFT模型在理论上能更科学地处理删失数据，是解决此类问题的黄金标准。")
    print("   > 建议采用AFT模型的预测结果 (AFT_predicted_weeks_by_BMI.csv) 进行后续的BMI分组与时点优化。")
else:
    print("【裁决】：反解GLM模型表现同样出色或更优！")
    print("   > 理由：反解GLM模型在误差指标上与AFT模型相当甚至更低，且与问题一的分析逻辑高度统一。")
    print("   > 建议采用反解GLM模型的预测结果进行后续分析，以保持整个解决方案的连贯性。")

# 保存两个模型的预测结果以供后续使用
df_survival.to_csv('model_predictions_comparison.csv', index=False)
print("\n两种模型的详细预测结果已保存至 'model_predictions_comparison.csv'")
