# -*- coding: utf-8 -*-
"""
@title: 女胎染色体异常判定模型 (XGBoost终极优化版)
@description: 这是针对问题四的最终优化方案。采纳了使用XGBoost替换随机森林的建议，
             旨在通过XGBoost强大的Boosting机制和内置的样本权重处理功能，
             显著提升对稀有但关键的“异常”样本的识别能力（召回率）。
"""

# ### 第一步：环境准备与数据加载
# ---
print("--- 开始第一步：环境准备与数据加载 ---")

# 1. 导入必要的库
# 基础数据处理
import pandas as pd
import numpy as np

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 数据预处理与模型评估
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# 【模型升级】导入XGBoost分类器
from xgboost import XGBClassifier

# 模型可解释性库
import shap


# ---
# 2. 全局设置 (确保图表能正确显示中文和负号)
# ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---
# 3. 加载数据
# ---
try:
    df_female = pd.read_csv('附件.xlsx - 女胎检测数据.csv')
    print("女胎检测数据加载成功！")
    print(f"数据集包含 {df_female.shape[0]} 条记录和 {df_female.shape[1]} 个字段。")
except FileNotFoundError:
    print("错误：未找到'附件.xlsx - 女胎检测数据.csv'文件。请检查文件路径。")
    exit()

# ### 第二步：数据预处理与特征工程
# ---
print("\n--- 开始第二步：数据预处理与特征工程 ---")

# 1. 目标变量处理
df_female['is_abnormal'] = np.where(df_female['染色体的非整倍体'].isnull(), 0, 1)
print("目标变量 'is_abnormal' 创建成功。")
print(df_female['is_abnormal'].value_counts())

# 2. 特征选择
feature_columns = [
    '年龄', '孕妇BMI', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
    'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', '被过滤掉读段数的比例',
    'X染色体浓度', '怀孕次数', '生产次数'
]

# 3. 数据清洗：处理非数值型数据
print("\n开始清洗特定列的非数值数据...")
for col in ['怀孕次数', '生产次数']:
    if df_female[col].dtype == 'object':
        df_female[col] = df_female[col].replace('≥3', '3')
        df_female[col] = pd.to_numeric(df_female[col], errors='coerce')
        print(f"'{col}' 列中的 '≥3' 已替换为 3，并已将该列转换为数值类型。")

# 4. 缺失值处理
print("\n开始处理缺失值...")
for col in feature_columns:
    if df_female[col].isnull().any():
        median_val = df_female[col].median()
        df_female[col] = df_female[col].fillna(median_val)
        print(f"'{col}' 列的缺失值已使用中位数 ({median_val:.2f}) 填充。")

X = df_female[feature_columns]
y = df_female['is_abnormal']
print("数据清洗与缺失值处理完成。")


# ### 第三步：实施分层诊断策略 (策略不变)
# ---
print("\n--- 开始第三步：实施分层诊断策略 ---")

# 1. 第一层：基于临床规则的快速筛查
Z_SCORE_THRESHOLD = 3
high_risk_condition = (
    (df_female['13号染色体的Z值'] > Z_SCORE_THRESHOLD) |
    (df_female['18号染色体的Z值'] > Z_SCORE_THRESHOLD) |
    (df_female['21号染色体的Z值'] > Z_SCORE_THRESHOLD)
)
df_high_risk = df_female[high_risk_condition]
df_to_diagnose = df_female[~high_risk_condition]
print(f"第一层规则筛查完成:")
print(f"  - 直接判定为高风险的样本数: {len(df_high_risk)}")
print(f"  - 需要送入机器学习模型进行精细诊断的样本数: {len(df_to_diagnose)}")

# 2. 准备并划分机器学习数据集
X_ml = df_to_diagnose[feature_columns]
y_ml = df_to_diagnose['is_abnormal']
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)
print(f"\n已将精细诊断数据集划分为训练集 ({len(X_train_ml)}条) 和测试集 ({len(X_test_ml)}条)。")


# ### 第四步：XGBoost模型训练 (核心优化部分)
# ---
print("\n--- 开始第四步：XGBoost模型训练 ---")
 
# 1. 【优化策略】使用 scale_pos_weight 处理样本不均衡
# 这是XGBoost内置的强大功能，直接告诉模型要更加关注少数类（异常样本）。
# 计算方法：多数类样本数 / 少数类样本数
counts = y_train_ml.value_counts()
scale_pos_weight_value = 2*counts[0] / counts[1]
print(f"训练集中 正常样本数(0): {counts[0]}, 异常样本数(1): {counts[1]}")
print(f"计算得到的 scale_pos_weight 值为: {scale_pos_weight_value:.2f}")
print("这个值将告诉模型，在计算损失时，一个异常样本的错误要比一个正常样本的错误严重得多。")

# 2. 特征标准化 (对XGBoost同样有益)
scaler = StandardScaler()
X_train_ml_scaled = scaler.fit_transform(X_train_ml)
X_test_ml_scaled = scaler.transform(X_test_ml)
print("\n特征标准化完成。")

# 3. 训练XGBoost模型
model_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight_value,
    objective='binary:logistic',
    n_estimators=100,
    random_state=42,
    use_label_encoder=False, 
    eval_metric='aucpr'  # 推荐使用PR曲线下面积，对不平衡数据更敏感
)

model_xgb.fit(X_train_ml_scaled, y_train_ml)
print("XGBoost模型训练完成！")

# ### 第五步：模型性能综合评估
# ---
print("\n--- 开始第五步：模型性能综合评估 ---")

# 1. 在测试集上进行预测

y_pred_proba_ml = model_xgb.predict_proba(X_test_ml_scaled)[:, 1]

THRESHOLD = 0.5  # 试试 0.25 ~ 0.4，根据召回率/精确率权衡

# 手动调整分类决策
y_pred_adjusted = (y_pred_proba_ml >= THRESHOLD).astype(int)


# 2. 打印分类报告 (重点关注召回率的变化)
print("--- XGBoost模型在测试集上的分类报告 ---")
print(classification_report(y_test_ml, y_pred_adjusted))

# 3. 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_ml, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测为正常', '预测为异常'],
            yticklabels=['实际为正常', '实际为异常'])
plt.title('XGBoost 混淆矩阵', fontsize=16)
plt.ylabel('实际类别', fontsize=12)
plt.xlabel('预测类别', fontsize=12)
plt.show()

# 4. 绘制ROC曲线和PR曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
# ROC 曲线
fpr, tpr, _ = roc_curve(y_test_ml, y_pred_proba_ml)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('假阳性率'); ax1.set_ylabel('真阳性率')
ax1.set_title('ROC 曲线'); ax1.legend(loc="lower right")
# PR 曲线
precision, recall, _ = precision_recall_curve(y_test_ml, y_pred_proba_ml)
ap = average_precision_score(y_test_ml, y_pred_proba_ml)
ax2.plot(recall, precision, color='blue', lw=2, label=f'PR 曲线 (AP = {ap:.2f})')
ax2.set_xlabel('召回率'); ax2.set_ylabel('精确率')
ax2.set_title('PR 曲线'); ax2.legend(loc="lower left")
plt.suptitle("XGBoost模型综合性能评估", fontsize=20)
plt.show()

# ### 第六步：模型可解释性分析 (使用SHAP)
# ---
print("\n--- 开始第六步：模型可解释性分析 ---")

# 1. 获取并展示特征重要性
importances = model_xgb.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': feature_columns,
    '重要性': importances
}).sort_values(by='重要性', ascending=False)
print("--- XGBoost模型认为最重要的特征排名 ---")
print(feature_importance_df)

# 2. 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='重要性', y='特征', data=feature_importance_df)
plt.title('XGBoost模型特征重要性', fontsize=16)
plt.show()

# 3. SHAP分析 (深入解释模型决策)
# TreeExplainer 适用于所有树模型，包括XGBoost
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_ml_scaled)

# 转换为DataFrame以便SHAP绘图时显示特征名称
X_test_df_for_shap = pd.DataFrame(X_test_ml_scaled, columns=feature_columns)

# 绘制SHAP特征影响图
shap.summary_plot(shap_values, X_test_df_for_shap, plot_type="bar", show=False)
plt.title('SHAP 特征重要性', fontsize=16)
plt.show()

shap.summary_plot(shap_values, X_test_df_for_shap, show=False)
plt.title('SHAP 特征影响详解图', fontsize=16)
plt.show()


print("\n--- 流程结束 ---")
