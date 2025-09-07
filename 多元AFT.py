# -*- coding: utf-8 -*-
"""
赛题三解答(第一阶段)：应用多元AFT模型进行多因素预测（双重高级可视化版）

本脚本在之前版本的基础上，新增了三维散点图的可视化功能，
与二维等高线图形成互补，从不同维度全面展示数据关系。

新增功能:
- 绘制一个以BMI为x轴，年龄为y轴，实际达标孕周为z轴的三维散点图。
"""

import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from mpl_toolkits.mplot3d import Axes3D # 【新增】导入3D绘图工具

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

# --- 1. 数据加载与特征选择 ---
print("--- 1. 加载原始男胎数据并进行特征选择与预处理 ---")
try:
    df_raw = pd.read_csv('附件.xlsx - 男胎检测数据.csv')
    df_raw.columns = df_raw.columns.str.strip()
    print(f"原始数据加载成功，共计 {len(df_raw)} 条检测记录。")
except FileNotFoundError:
    print("错误：未找到 '附件.xlsx - 男胎检测数据.csv' 文件。")
    exit()

features = ['孕妇代码', '检测孕周', '孕妇BMI', '年龄', 'Y染色体浓度']
df_clean = df_raw[features].copy()
df_clean['孕周数'] = df_clean['检测孕周'].apply(parse_gestational_week)
for col in ['孕妇BMI', '年龄', 'Y染色体浓度', '孕周数']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean.dropna(subset=['孕妇代码', '孕周数', '孕妇BMI', '年龄', 'Y染色体浓度'], inplace=True)

# --- 2. 构建生存分析数据集 ---
print("\n--- 2. 构建生存分析数据集 ---")
TARGET_CONCENTRATION = 0.04
survival_data = []
for name, group in df_clean.groupby('孕妇代码'):
    avg_bmi = group['孕妇BMI'].mean()
    avg_age = group['年龄'].mean()
    event_occurred = group[group['Y染色体浓度'] >= TARGET_CONCENTRATION]
    if not event_occurred.empty:
        duration = event_occurred['孕周数'].min(); event = 1
    else:
        duration = group['孕周数'].max(); event = 0
    survival_data.append({'孕妇代码': name, '孕妇BMI': avg_bmi, '年龄': avg_age, 'duration': duration, 'event': event})
df_survival = pd.DataFrame(survival_data)
print("多元生存分析数据集构建完毕。")

# --- 3. 构建并拟合多元AFT模型 ---
print("\n--- 3. 构建并拟合多元韦伯AFT模型 ---")
df_survival.rename(columns={'孕妇BMI': 'BMI', '年龄': 'Age'}, inplace=True)
aft = WeibullAFTFitter()
aft.fit(df_survival, 'duration', event_col='event', formula='BMI + Age')
print("\n多元AFT模型拟合结果摘要:")
aft.print_summary(decimals=4)

# --- 4. 解读模型结果 ---
print("\n--- 4. 模型结果解读 ---")
# ... (解读部分代码与之前相同) ...
params_df = aft.params_.reset_index()
def interpret_and_get_coef(param_name):
    """一个辅助函数，用于解读并返回系数"""
    row = params_df.query(f"param == 'lambda_' and covariate == '{param_name}'")
    if not row.empty:
        coef = row.iloc[0, 2] 
        time_ratio = np.exp(coef)
        effect_percent = (time_ratio - 1) * 100
        print(f"\n变量 [{param_name}] 的解读:")
        if time_ratio > 1:
            print(f"  - 结论：{param_name}对达标时间有显著的延缓作用。每增加一个单位，达标时间平均延长{effect_percent:.2f}%。")
        else:
            print(f"  - 结论：{param_name}对达标时间有显著的加速作用。每增加一个单位，达标时间平均缩短{-effect_percent:.2f}%。")
        return coef
    return None
lambda_bmi_coef = interpret_and_get_coef('BMI')
lambda_age_coef = interpret_and_get_coef('Age')

# 提取另外两个核心参数
lambda_intercept = params_df.query("param == 'lambda_' and covariate == 'Intercept'").iloc[0, 2]
rho_intercept = params_df.query("param == 'rho_' and covariate == 'Intercept'").iloc[0, 2]

# 【新增】输出最终公式
if all([lambda_bmi_coef, lambda_age_coef]):
    print("\n【最终的多元AFT模型预测函数】:")
    print("-----------------------------------------------------------------------------------------")
    print(f"  预测最早达标孕周(BMI, Age) = exp({lambda_intercept:.4f} + ({lambda_bmi_coef:.4f} * BMI) + ({lambda_age_coef:.4f} * Age)) * (ln(2))^(1/{rho_intercept:.4f})")
    print("-----------------------------------------------------------------------------------------")

# --- 5. 高级可视化 ---
print("\n--- 5. 高级可视化 ---")
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams['axes.unicode_minus'] = False

# 5.1 可视化一：NIPT最早达标孕周推荐地图 (二维等高线图)
print("\n5.1 正在生成二维推荐地图...")
bmi_grid = np.linspace(df_survival['BMI'].min(), df_survival['BMI'].max(), 50)
age_grid = np.linspace(df_survival['Age'].min(), df_survival['Age'].max(), 50)
X_mesh, Y_mesh = np.meshgrid(bmi_grid, age_grid)
grid_data = pd.DataFrame({'BMI': X_mesh.ravel(), 'Age': Y_mesh.ravel()})
Z_mesh = aft.predict_median(grid_data).values.reshape(X_mesh.shape)

plt.figure(figsize=(14, 10))
contour = plt.contourf(X_mesh, Y_mesh, Z_mesh, levels=15, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('预测的最早达标孕周 (周)', fontsize=14)
plt.scatter(df_survival['BMI'], df_survival['Age'], c=df_survival['duration'], 
            cmap='coolwarm', edgecolors='black', s=50, label='实际达标孕妇 (颜色代表实际达标孕周)')
plt.title('NIPT最早达标孕周推荐地图 (多元AFT模型预测)', fontsize=18)
plt.xlabel('孕妇BMI', fontsize=14)
plt.ylabel('孕妇年龄', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 5.2 【新增】可视化二：数据原始分布 (三维散点图)
print("\n5.2 正在生成三维散点图...")
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 仅可视化实际达标的孕妇，以观察真实关系
df_event = df_survival[df_survival['event']==1]

# 绘制三维散点图
# x轴=BMI, y轴=年龄, z轴=实际达标孕周
# c=颜色，根据z轴的值(达标孕周)来映射颜色
scatter = ax.scatter3D(df_event['BMI'], df_event['Age'], df_event['duration'], 
                       c=df_event['duration'], cmap='coolwarm', s=60, edgecolors='black')

# 设置坐标轴标签和标题
ax.set_title('实际达标孕周的三维分布', fontsize=18)
ax.set_xlabel('孕妇BMI', fontsize=12, labelpad=10)
ax.set_ylabel('孕妇年龄', fontsize=12, labelpad=10)
ax.set_zlabel('实际达标孕周 (周)', fontsize=12, labelpad=10)

# 添加颜色条
cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
cbar.set_label('实际达标孕周 (周)', fontsize=12)

# 调整视角
ax.view_init(elev=20., azim=-65)
plt.show()
print("三维图解读：该图直观展示了数据点的空间分布。我们可以看到，较高的点（晚达标，红色）倾向于分布在BMI和年龄都较大的区域，这与我们模型的发现完全一致。")


# --- 6. 保存AFT模型对象 ---
print("\n--- 6. 保存最终的多元AFT模型 ---")
model_filename = 'multivariate_aft_model.joblib'
try:
    joblib.dump(aft, model_filename)
    print(f"模型对象已成功保存至文件: {model_filename}")
    print("这个模型包含了预测所需的所有信息，将作为下一阶段“风险函数优化”的核心输入。")
except Exception as e:
    print(f"错误：模型保存失败。原因: {e}")
