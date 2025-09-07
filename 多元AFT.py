# -*- coding: utf-8 -*-
"""
NIPT最佳检测时点推荐：基于多策略AFT生存模型。

本脚本利用韦伯加速失效时间（AFT）模型，分析孕妇BMI对其NIPT检测
Y染色体浓度达标时间的影响。

核心功能：
- 构建基于删失数据的生存分析数据集。
- 训练Weibull AFT模型。
- 预测在不同置信度（如50%, 75%, 90%）下，不同BMI孕妇的建议检测孕周。
- 可视化展示不同策略下的推荐时点曲线，为临床决策提供依据。
"""
import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt
import seaborn as sns

def parse_gestational_week(week_str):
    """将 '12w+3' 或 '12w' 格式的孕周字符串转换为数值。"""
    if not isinstance(week_str, str):
        return np.nan
    week_str = week_str.strip()
    try:
        if 'w+' in week_str:
            parts = week_str.split('w+')
            return int(parts[0]) + int(parts[1]) / 7.0
        elif 'w' in week_str:
            return float(week_str.split('w')[0])
        return np.nan
    except (ValueError, IndexError):
        return np.nan

def prepare_survival_data(raw_data_path, target_concentration=0.04):
    """加载原始数据并构建用于生存分析的数据集。"""
    try:
        df_raw = pd.read_csv(raw_data_path)
        df_raw.columns = df_raw.columns.str.strip()
    except FileNotFoundError:
        print(f"错误：原始数据文件未找到于 '{raw_data_path}'")
        return None
    
    df_clean = df_raw[['孕妇代码', '检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
    df_clean['孕周数'] = df_clean['检测孕周'].apply(parse_gestational_week)
    for col in ['孕妇BMI', 'Y染色体浓度', '孕周数']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.dropna(inplace=True)

    survival_data = []
    for name, group in df_clean.groupby('孕妇代码'):
        event_observed = group[group['Y染色体浓度'] >= target_concentration]
        if not event_observed.empty:
            duration = event_observed['孕周数'].min()
            event = 1
        else:
            duration = group['孕周数'].max()
            event = 0
        survival_data.append({
            '孕妇BMI': group['孕妇BMI'].mean(),
            'duration': duration,
            'event': event
        })
    return pd.DataFrame(survival_data)

def train_and_predict_aft(df_survival):
    """训练AFT模型并基于不同达标率进行预测。"""
    df_survival.rename(columns={'孕妇BMI': 'BMI'}, inplace=True)
    aft = WeibullAFTFitter()
    aft.fit(df_survival, 'duration', event_col='event', formula='BMI')
    print("AFT模型拟合成功。")

    bmi_range = np.linspace(df_survival['BMI'].min(), df_survival['BMI'].max(), 200)
    df_bmi_range = pd.DataFrame({'BMI': bmi_range})

    # 定义不同的达标率策略
    pass_rates = {
        "50%达标率 (中位数)": 0.50,
        "75%达标率": 0.75,
        "90%达标率 (高置信度)": 0.90,
        "95%达标率 (极高置信度)": 0.95
    }
    
    df_predictions = df_bmi_range.copy()
    for label, rate in pass_rates.items():
        # AFT的predict_percentile的p代表的是失效概率，所以 p = 1 - 达标率
        predicted_weeks = aft.predict_percentile(df_bmi_range, p=(1 - rate))
        df_predictions[label] = predicted_weeks
        
    return df_predictions, pass_rates

def visualize_multistrategy_results(df_predictions, df_survival, pass_rates):
    """可视化多策略的NIPT推荐时点。"""
    sns.set_theme(style="whitegrid", font="SimHei")
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(16, 10))

    sns.scatterplot(data=df_survival[df_survival['event']==1], x='BMI', y='duration',
                    color='lightgray', alpha=0.8, label='实际观测到的达标孕妇')

    colors = ['red', 'green', 'blue', 'purple']
    styles = ['-', '--', '-.', ':']
    for i, label in enumerate(pass_rates.keys()):
        plt.plot(df_predictions['BMI'], df_predictions[label], color=colors[i],
                 linestyle=styles[i], linewidth=3, label=f'预测曲线 ({label})')

    plt.title('NIPT推荐时点决策图：不同达标率策略对比', fontsize=20)
    plt.xlabel('孕妇BMI', fontsize=14)
    plt.ylabel('推荐的NIPT检测时点 (周)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    """主执行函数。"""
    raw_data_path = '附件.xlsx - 男胎检测数据.csv'
    output_filename = 'AFT_multistrategy_predictions.csv'

    print("--- 开始构建生存分析数据集 ---")
    df_survival = prepare_survival_data(raw_data_path)
    
    if df_survival is not None:
        print(f"数据集构建成功，共 {len(df_survival)} 位孕妇信息。")
        print("\n--- 开始训练AFT模型并进行多策略预测 ---")
        df_predictions, pass_rates = train_and_predict_aft(df_survival)
        
        df_predictions.to_csv(output_filename, index=False)
        print(f"\n多策略预测结果已保存至: {output_filename}")
        
        print("\n--- 开始生成可视化决策图 ---")
        visualize_multistrategy_results(df_predictions, df_survival, pass_rates)
        print("流程结束。")

if __name__ == '__main__':
    main()