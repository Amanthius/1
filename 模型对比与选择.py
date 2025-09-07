# -*- coding: utf-8 -*-
"""
问题一模型对比附录：最终公式汇总

本脚本在“问题一终极模型探索”的基础上进行了关键修改。
它不仅完整地执行了多个高级模型的系统性比较，更在最后新增了一个
“模型对比附录”环节，将所有关键候选模型的最终数学公式进行统一输出，
以供您在论文中进行全面的对比分析。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- 1. 数据加载与特征工程 ---
print("--- 1. 加载数据并进行特征工程 ---")
try:
    df = pd.read_excel('NIPT_Data_Cleaned.xlsx', sheet_name='处理后的男胎数据')
    model_df = df[['Y染色体浓度', '孕周数', '孕妇BMI']].copy().dropna()
    print(f"原始数据加载成功，共计 {len(model_df)} 条记录。")
except FileNotFoundError:
    print("错误：未找到 'NIPT_Data_Cleaned.xlsx' 文件。")
    exit()

# 1.1 创建对数转换后的因变量
# Y染色体浓度有0或接近0的值，直接取log会出错，因此加上一个极小量
model_df['log_Y_conc'] = np.log(model_df['Y染色体浓度'] + 1e-9)

# 1.2 创建中心化的变量 (变量 - 均值) 以优化交互项
model_df['孕周数_cen'] = model_df['孕周数'] - model_df['孕周数'].mean()
model_df['孕妇BMI_cen'] = model_df['孕妇BMI'] - model_df['孕妇BMI'].mean()

# 1.3 基于中心化变量创建交互项
model_df['孕周数_x_BMI_cen'] = model_df['孕周数_cen'] * model_df['孕妇BMI_cen']
print("特征工程完毕。")

# --- 2. 构建并系统比较多个优化模型 ---
print("\n--- 2. 构建并系统比较多个优化模型 ---")

# 准备一个字典来存储所有模型的结果
results = {}

# --- 模型A: 基准简单线性模型 (OLS) ---
y = model_df['Y染色体浓度']
X_simple = sm.add_constant(model_df[['孕周数', '孕妇BMI']])
model_simple = sm.OLS(y, X_simple).fit()
results['OLS_Simple'] = {'model': model_simple, 'AIC': model_simple.aic}

# --- 模型B: 带中心化交互项的线性模型 (OLS) ---
X_interaction = sm.add_constant(model_df[['孕周数_cen', '孕妇BMI_cen', '孕周数_x_BMI_cen']])
model_interaction = sm.OLS(y, X_interaction).fit()
results['OLS_Interaction'] = {'model': model_interaction, 'AIC': model_interaction.aic}

# --- 模型C: 对数-线性模型 (在log(Y)上回归) ---
y_log = model_df['log_Y_conc']
X_log = sm.add_constant(model_df[['孕周数', '孕妇BMI']])
model_log = sm.OLS(y_log, X_log).fit()
results['OLS_LogY'] = {'model': model_log, 'AIC': model_log.aic}

# --- 模型D: Gamma广义线性模型 (GLM) ---
model_glm = sm.GLM(y, X_simple, family=sm.families.Gamma(link=sm.families.links.log())).fit()
results['GLM_Gamma'] = {'model': model_glm, 'AIC': model_glm.aic}

# 打印所有模型的AIC进行比较
print("\n各模型AIC值对比 (越低越好):")
for name, res in sorted(results.items(), key=lambda item: item[1]['AIC']):
    print(f"  - {name}: {res['AIC']:.2f}")

# 根据AIC值找到最佳模型
best_model_name = min(results, key=lambda k: results[k]['AIC'])
print(f"\nAIC对比结果显示，最佳模型是: 【{best_model_name}】")


# --- 附录：【新增】所有候选模型最终公式汇总 ---
print("\n" + "="*80)
print("--- 附录：问题一候选模型最终公式汇总 ---")
print("="*80)

# 模型A: 简单线性模型 (OLS)
p = model_simple.params
print("\n【模型A: 简单线性模型 (OLS)】")
print("-----------------------------------------------------------------")
print(f"  E[Y_conc] = {p['const']:.4f} + ({p['孕周数']:.4f} * GA) + ({p['孕妇BMI']:.4f} * BMI)")
print("-----------------------------------------------------------------")

# 模型B: 交互项线性模型 (OLS_Interaction)
# 注意：这里的GA和BMI是中心化后的变量
p = model_interaction.params
print("\n【模型B: 交互项线性模型 (OLS_Interaction)】")
print("-----------------------------------------------------------------")
print(f"  E[Y_conc] = {p['const']:.4f} + ({p['孕周数_cen']:.4f} * GA_cen) + ({p['孕妇BMI_cen']:.4f} * BMI_cen) + ({p['孕周数_x_BMI_cen']:.4f} * GA_cen * BMI_cen)")
print("  (注: GA_cen = GA - 平均孕周, BMI_cen = BMI - 平均BMI)")
print("-----------------------------------------------------------------")

# 模型C: 对数-线性模型 (OLS_LogY)
p = model_log.params
print("\n【模型C: 对数-线性模型 (OLS_LogY)】")
print("-----------------------------------------------------------------")
print(f"  ln(E[Y_conc]) = {p['const']:.4f} + ({p['孕周数']:.4f} * GA) + ({p['孕妇BMI']:.4f} * BMI)")
print(f"  E[Y_conc] = exp({p['const']:.4f} + ({p['孕周数']:.4f} * GA) + ({p['孕妇BMI']:.4f} * BMI))")
print("-----------------------------------------------------------------")
    
# 模型D: Gamma广义线性模型 (GLM_Gamma) - 问题一最优解
p = model_glm.params
print("\n【模型D: Gamma广义线性模型 (GLM_Gamma) - 问题一最优解】")
print("-----------------------------------------------------------------")
print(f"  ln(E[Y_conc]) = {p['const']:.4f} + ({p['孕周数']:.4f} * GA) + ({p['孕妇BMI']:.4f} * BMI)")
print(f"  E[Y_conc] = exp({p['const']:.4f} + ({p['孕周数']:.4f} * GA) + ({p['孕妇BMI']:.4f} * BMI))")
print("-----------------------------------------------------------------")

print("\n(Y_conc: Y染色体浓度, GA: 孕周数, BMI: 孕妇身体质量指数)")
