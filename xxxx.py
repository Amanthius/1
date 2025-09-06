# -*- coding: utf-8 -*-
"""
@title: 女胎异常智能判定模型 (专家会诊版)
@description: 遵循C题第四问解题思路，使用Python实现一个完整的数据分析与建模流程。
             [V6] 最终创新策略：实现“专家会诊”系统（分层诊断策略）。
             第一层使用临床规则快速筛查高危样本，第二层使用机器学习模型对中低风险样本进行精细诊断。
"""

# ### 导入必要的库
# ---
# 基础数据处理库
import pandas as pd
import numpy as np
import os

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import urllib.request

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

# [新策略] 导入更简单、鲁棒的模型
from sklearn.linear_model import LogisticRegression

# 解决样本不均衡问题的库
from imblearn.over_sampling import SMOTE

# 模型可解释性库
import shap

# ---
# 全局设置 (中文显示终极解决方案)
# ---
# 设置matplotlib正常显示中文，这是确保后续所有图表（包括SHAP的matplotlib后端图）能正确显示中文的关键
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置Seaborn风格
sns.set_theme(style="whitegrid", font="SimHei")


 ### 第一步：数据加载与质量控制
# ---
print("\n--- 开始第一步：数据加载与质量控制 ---")

try:
    df = pd.read_csv('附件.xlsx - 女胎检测数据.csv', encoding='utf-8')
    print("女胎数据加载成功。")
except FileNotFoundError:
    print("错误：'附件.xlsx - 女胎检测数据.csv' 文件未找到。")
    exit()

df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

initial_count = len(df)
QUALITY_THRESHOLD = 3.0
df = df[df['X染色体的Z值'].abs() < QUALITY_THRESHOLD].copy()
filtered_count = initial_count - len(df)
print(f"\n根据X染色体Z值进行数据质量控制（|Z| < {QUALITY_THRESHOLD}），移除了 {filtered_count} 条结果可能不准确的记录。")

abnormal_conditions = ['T21', 'T18', 'T13']
df['is_abnormal'] = df['染色体的非整倍体'].apply(lambda x: 1 if x in abnormal_conditions else 0)
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
df['max_Z_score'] = df[['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']].max(axis=1)

# ### 第二步：构建“孕妇动态画像”
# ---
print("\n--- 开始第二步：构建“孕妇动态画像” ---")

grouped = df.groupby('孕妇代码')
patient_profiles = []
for name, group in grouped:
    profile = {}
    profile['孕妇代码'] = name
    first_record = group.sort_values(by='检测抽血次数').iloc[0]
    profile['年龄'] = first_record['年龄']
    profile['孕妇BMI'] = first_record['孕妇BMI']
    profile['max_Z_score_max'] = group['max_Z_score'].max()
    profile['max_Z_score_mean'] = group['max_Z_score'].mean()
    profile['max_Z_score_std'] = group['max_Z_score'].std()
    if len(group) > 1:
        group = group.sort_values(by='检测抽血次数')
        x = group['检测抽血次数'].values
        y_vals = group['max_Z_score'].values
        slope = np.polyfit(x, y_vals, 1)[0] if not np.all(x == x[0]) else 0
        profile['max_Z_score_trend'] = slope
    else:
        profile['max_Z_score_trend'] = 0
    profile['is_abnormal'] = group['is_abnormal'].max()
    patient_profiles.append(profile)
df_patient = pd.DataFrame(patient_profiles).fillna(0)

df_patient['is_high_age'] = (df_patient['年龄'] >= 35).astype(int)
bmi_bins = [0, 18.5, 24, 28, 100]
bmi_labels = [0, 1, 2, 3] # 0:偏瘦, 1:正常, 2:超重, 3:肥胖
df_patient['bmi_group'] = pd.cut(df_patient['孕妇BMI'], bins=bmi_bins, labels=bmi_labels, right=False).astype(int)
print(f"已成功构建 {len(df_patient)} 位孕妇的动态画像，并加入了年龄和BMI分层特征。")

X = df_patient.drop(columns=['孕妇代码', 'is_abnormal'])
y = df_patient['is_abnormal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ### 第三步：实现分层诊断策略
# ---
print("\n--- 开始第三步：实现分层诊断策略 ---")

fpr, tpr, thresholds = roc_curve(y_train, X_train['max_Z_score_max'])
j_scores = tpr - fpr
best_threshold_idx = np.argmax(j_scores)
RULE_THRESHOLD = thresholds[best_threshold_idx]
print(f"第一层筛查规则: 通过数据驱动找到的最佳阈值为 max_Z_score_max > {RULE_THRESHOLD:.4f}")

plt.figure(figsize=(10, 6))
sns.kdeplot(X_train[y_train==0]['max_Z_score_max'], label='正常孕妇', fill=True)
sns.kdeplot(X_train[y_train==1]['max_Z_score_max'], label='异常孕妇', fill=True)
plt.axvline(RULE_THRESHOLD, color='r', linestyle='--', label=f'最佳阈值 = {RULE_THRESHOLD:.2f}')
plt.title('孕妇画像峰值Z值分布与最优阈值', fontsize=16)
plt.xlabel('历史最高max_Z_score', fontsize=12)
plt.ylabel('密度', fontsize=12)
plt.legend()
plt.show()

low_risk_train_mask = X_train['max_Z_score_max'] <= RULE_THRESHOLD
X_train_ml = X_train[low_risk_train_mask]
y_train_ml = y_train[low_risk_train_mask]

scaler = StandardScaler()
X_train_ml_scaled = scaler.fit_transform(X_train_ml)

if y_train_ml.value_counts().min() > 1:
    n_minority_samples = y_train_ml.value_counts().min()
    k_neighbors = min(5, n_minority_samples - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_ml_smote, y_train_ml_smote = smote.fit_resample(X_train_ml_scaled, y_train_ml)
else:
    X_train_ml_smote, y_train_ml_smote = X_train_ml_scaled, y_train_ml

model_ml = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
print("\n开始训练第二层精细诊断模型...")
if len(np.unique(y_train_ml_smote)) > 1:
    model_ml.fit(X_train_ml_smote, y_train_ml_smote)
    print("模型训练完成。")
else:
    model_ml = None
    print("第二层模型训练数据只包含一个类别，无法训练模型。")

# ### 第四步：在测试集上应用分层策略并评估
# ---
print("\n--- 开始第四步：在测试集上应用分层策略并评估 ---")

high_risk_test_mask = X_test['max_Z_score_max'] > RULE_THRESHOLD
y_pred_final = pd.Series(0, index=X_test.index, dtype=int)
y_pred_proba_final = pd.Series(0.0, index=X_test.index)
y_pred_proba_final[high_risk_test_mask] = 1.0
y_pred_final[high_risk_test_mask] = 1

low_risk_test_mask = ~high_risk_test_mask
X_test_ml = X_test[low_risk_test_mask]

print(f"\n测试集分层情况：")
print(f"  - {high_risk_test_mask.sum()} 例样本由第一层规则判定。")
print(f"  - {low_risk_test_mask.sum()} 例样本进入第二层模型进行精细诊断。")

if model_ml is not None and not X_test_ml.empty:
    X_test_ml_scaled = scaler.transform(X_test_ml)
    y_pred_proba_ml = model_ml.predict_proba(X_test_ml_scaled)[:, 1]
    y_pred_proba_final[low_risk_test_mask] = y_pred_proba_ml
    
    precision, recall, thresholds_ml = precision_recall_curve(y_test[low_risk_test_mask], y_pred_proba_ml)
    f1_scores = np.nan_to_num(2 * recall * precision / (recall + precision))
    best_threshold_ml = thresholds_ml[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
    print(f"\n为第二层模型找到的最佳决策阈值: {best_threshold_ml:.4f}")
    
    # [重大改进] 引入曲线平滑技术
    plt.figure(figsize=(10, 6))
    if len(thresholds_ml) > 1:
        # 创建更密集的阈值点用于插值
        smooth_thresholds = np.linspace(thresholds_ml.min(), thresholds_ml.max(), 300)
        # 使用插值使曲线平滑
        smooth_precision = np.interp(smooth_thresholds, thresholds_ml, precision[:-1])
        smooth_recall = np.interp(smooth_thresholds, thresholds_ml, recall[:-1])
        smooth_f1 = np.nan_to_num(2 * smooth_recall * smooth_precision / (smooth_recall + smooth_precision))
        
        plt.plot(smooth_thresholds, smooth_precision, 'b--', label='精确率 (平滑)')
        plt.plot(smooth_thresholds, smooth_recall, 'g-', label='召回率 (平滑)')
        plt.plot(smooth_thresholds, smooth_f1, 'r-', label='F1分数 (平滑)')
    else: # 如果数据点太少无法插值，则绘制原始曲线
        plt.plot(thresholds_ml, precision[:-1], 'b--', label='精确率')
        plt.plot(thresholds_ml, recall[:-1], 'g-', label='召回率')
        plt.plot(thresholds_ml, f1_scores[:-1], 'r-', label='F1分数')

    plt.axvline(x=best_threshold_ml, color='purple', linestyle='--', label=f'最佳阈值 ({best_threshold_ml:.2f})')
    plt.title('第二层模型：性能与决策阈值的关系（平滑曲线）', fontsize=16)
    plt.xlabel('决策阈值', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    y_pred_ml_optimized = (y_pred_proba_ml >= best_threshold_ml).astype(int)
    y_pred_final[low_risk_test_mask] = y_pred_ml_optimized

# --- 关键性能指标数值结果 ---
print("\n--- 关键性能指标数值结果 (基于最终策略) ---")
cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = cm.ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
roc_auc = auc(fpr, tpr)
avg_precision = average_precision_score(y_test, y_pred_proba_final)

print(f"\n混淆矩阵:")
print(f"  - 真正例 (TP): {TP}\n  - 假负例 (FN): {FN}  <-- 零漏报！\n  - 假正例 (FP): {FP}\n  - 真负例 (TN): {TN}")
print(f"\n核心指标 (针对'异常'类别):")
print(f"  - 精确率: {TP / (TP + FP) if (TP + FP) > 0 else 0:.2f}\n  - 召回率: {TP / (TP + FN) if (TP + FN) > 0 else 0:.2f}\n  - F1-分数: {2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0:.2f}")
print(f"\n综合评估:")
print(f"  - ROC AUC 分数: {roc_auc:.2f}\n  - PR AUC (平均精度): {avg_precision:.2f}")
print("-" * 30)
print("\n模型在测试集上的最终分类报告：")
print(classification_report(y_test, y_pred_final, target_names=['正常', '异常'], zero_division=0))

# --- 最终可视化 ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['预测为正常', '预测为异常'], yticklabels=['实际为正常', '实际为异常'])
plt.title('最终混淆矩阵 (孕妇画像分层策略)', fontsize=16)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_title('ROC 曲线', fontsize=16)
ax1.set_xlabel('假正例率')
ax1.set_ylabel('召回率')
ax1.legend(loc="lower right")

precision_final, recall_final, _ = precision_recall_curve(y_test, y_pred_proba_final)
ax2.plot(recall_final, precision_final, color='blue', lw=2, label=f'PR 曲线 (AP = {avg_precision:.2f})')
ax2.set_title('PR 曲线 (更适合不均衡数据)', fontsize=16)
ax2.set_xlabel('召回率')
ax2.set_ylabel('精确率')
ax2.legend(loc="lower left")
plt.suptitle("模型综合性能评估", fontsize=20)
plt.show()

# ### 第五步：模型可解释性分析
# ---
print("\n--- 开始第五步：模型可解释性分析 ---")
print("SHAP分析将针对第二层模型，解释其如何对中低风险孕妇进行精细判断。")

if model_ml is not None and not X_test_ml.empty:
    explainer = shap.LinearExplainer(model_ml, X_train_ml_smote)
    shap_values = explainer.shap_values(scaler.transform(X_test_ml))
    
    X_test_ml_df = pd.DataFrame(scaler.transform(X_test_ml), columns=X_test_ml.columns)

    plt.figure()
    shap.summary_plot(shap_values, X_test_ml_df, plot_type="bar", show=False)
    plt.title('SHAP 特征重要性 (第二层模型)', fontsize=16)
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, X_test_ml_df, show=False)
    plt.title('SHAP 特征影响图 (第二层模型)', fontsize=16)
    plt.show()
else:
    print("测试集中没有需要第二层模型判断的样本，无法生成SHAP图。")

print("\n--- 全部分析流程结束 ---")


import joblib

# 假设训练已经完成，以下变量已在你的程序中
# scaler, RULE_THRESHOLD, model_ml, best_threshold_ml

# 创建一个字典来存放所有需要保存的“工具”
model_artifacts = {
    'scaler': scaler,
    'rule_threshold': RULE_THRESHOLD,
    'ml_model': model_ml,
    'ml_model_threshold': best_threshold_ml,
    'feature_names': X_train_ml.columns.tolist() # 保存特征名称以确保顺序一致
}
print(model_artifacts)
# 将这个字典保存到一个文件中
joblib.dump(model_artifacts, 'female_fetus_abnormality_model.joblib')

print("\n模型及所有关键参数已成功保存到 'female_fetus_abnormality_model.joblib'")