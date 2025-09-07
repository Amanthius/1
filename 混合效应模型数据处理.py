# -*- coding: utf-8 -*-
"""
生成清洗后数据的最终脚本

本脚本读取原始的、未经处理的CSV文件，执行一套完整的数据清洗、
标准化和整合流程，最终生成用于所有后续分析的
'NIPT_Data_Cleaned_Final.xlsx' 文件。

核心功能：
- 标准化孕周格式。
- 清洗异常值与缺失值。
- 【关键】使用混合效应模型科学地整合同一男胎孕妇的多次检测记录。
- 为女胎数据创建分类任务所需的目标变量。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

# --- 配置区域 ---
warnings.filterwarnings("ignore", category=Warning)

# --- 辅助函数定义 ---
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

def aggregate_male_data(group):
    """
    健壮地聚合单个男胎孕妇的多次检测数据。
    """
    if len(group) == 1: return group.iloc[0]
    y = group['Y染色体浓度']; X = sm.add_constant(group['孕周数'])
    try:
        md = MixedLM(endog=y, exog=X, groups=np.ones(len(group))); mdf = md.fit(reml=False)
        mean_week = group['孕周数'].mean()
        pred_conc = mdf.predict(pd.DataFrame({'const': 1, '孕周数': [mean_week]})).iloc[0]
        result = group.iloc[0].copy()
        numeric_cols = group.select_dtypes(include=np.number).columns
        result[numeric_cols] = group[numeric_cols].mean(); result['Y染色体浓度'] = pred_conc
        return result
    except Exception: return group.mean(numeric_only=True)

# --- 主清洗流程 ---
def run_cleaning_pipeline(male_file, female_file, output_file):
    print("--- 开始执行NIPT数据清洗流程 ---")

    # 1. 加载数据
    try:
        df_male = pd.read_csv(male_file); df_female = pd.read_csv(female_file)
        print("✅ 1/5: 原始数据加载成功。")
    except FileNotFoundError:
        print("错误：找不到数据文件。"); return

    # 2. 清洗男胎数据
    print("⏳ 2/5: 开始处理男胎数据...")
    df_male.columns = df_male.columns.str.strip()
    df_male['孕周数'] = df_male['检测孕周'].apply(parse_gestational_week)
    male_cols = ['孕妇代码', '年龄', '身高', '体重', '孕妇BMI', '孕周数', 'Y染色体浓度']
    df_male_cleaned = df_male[male_cols].copy()
    for col in male_cols[1:]: df_male_cleaned[col] = pd.to_numeric(df_male_cleaned[col], errors='coerce')
    
    df_male_cleaned.dropna(inplace=True)
    df_male_cleaned = df_male_cleaned[
        (df_male_cleaned['年龄'].between(15, 55)) & (df_male_cleaned['孕妇BMI'].between(15, 60)) &
        (df_male_cleaned['孕周数'].between(10, 42)) & (df_male_cleaned['Y染色体浓度'].between(0, 1))
    ]
    
    print("⏳ 3/5: 开始用健壮模式整合男胎数据...")
    processed_rows = []
    for mother_id, group in df_male_cleaned.groupby('孕妇代码'):
        aggregated_row = aggregate_male_data(group)
        processed_rows.append(aggregated_row)
    df_male_final = pd.DataFrame(processed_rows)
    print(f"✅ 男胎数据整合完毕，最终得到 {len(df_male_final)} 位孕妇的独立数据。")

    # 4. 清洗女胎数据
    print("⏳ 4/5: 开始处理女胎数据...")
    df_female.columns = df_female.columns.str.strip()
    df_female['孕周数'] = df_female['检测孕周'].apply(parse_gestational_week)
    female_cols = ['孕妇代码', '年龄', '孕妇BMI', '孕周数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度', '染色体的非整倍体']
    df_female_cleaned = df_female[female_cols].copy()
    for col in df_female_cleaned.columns:
        if col not in ['孕妇代码', '染色体的非整倍体']: df_female_cleaned[col] = pd.to_numeric(df_female_cleaned[col], errors='coerce')
    df_female_cleaned['is_abnormal'] = df_female_cleaned['染色体的非整倍体'].apply(lambda x: 1 if isinstance(x, str) and any(ab in x for ab in ['T21', 'T18', 'T13']) else 0)
    df_female_cleaned.dropna(subset=['孕妇BMI', '孕周数', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值'], inplace=True)
    df_female_cleaned = df_female_cleaned[
        (df_female_cleaned['年龄'].between(15, 55)) & (df_female_cleaned['孕妇BMI'].between(15, 60)) & (df_female_cleaned['孕周数'].between(10, 42))
    ]
    print(f"✅ 女胎数据清洗完毕，共 {len(df_female_cleaned)} 条记录可用于建模。")

    # 5. 保存结果
    print("\n--- 5. 保存最终清洗结果 ---")
    try:
        with pd.ExcelWriter(output_file) as writer:
            df_male_final.to_excel(writer, sheet_name='处理后的男胎数据', index=False)
            df_female_cleaned.to_excel(writer, sheet_name='处理后的女胎数据', index=False)
        print(f"🎉🎉🎉 清洗流程结束，结果已成功保存至文件 -> {output_file}")
    except Exception as e:
        print(f"❌ 文件保存失败: {e}")

# --- 程序主入口 ---
if __name__ == '__main__':
    MALE_DATA_FILE = '附件.xlsx - 男胎检测数据.csv'
    FEMALE_DATA_FILE = '附件.xlsx - 女胎检测数据.csv'
    OUTPUT_FILE = 'NIPT_Data_Cleaned.xlsx'
    
    run_cleaning_pipeline(MALE_DATA_FILE, FEMALE_DATA_FILE, OUTPUT_FILE)

