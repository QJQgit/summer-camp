# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv('data//train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# === 年龄缺失值处理模块（方案1：按头衔分组填充） ===
# 1. 提取头衔（如 Mr/Mrs/Miss/Master 等）
def extract_title(name):
    title = name.split(', ')[1].split('. ')[0]
    return title
 
df['Title'] = df['Name'].apply(extract_title)
 
# 2. 合并稀有头衔（避免分组过多）
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].apply(lambda x: 'Other' if x in rare_titles else x)
 
# 3. 按头衔分组计算年龄中位数，并填充缺失值
title_age_median = df.groupby('Title')['Age'].transform('median')
df['Age'].fillna(title_age_median, inplace=True)
 
# 4. 删除临时创建的 'Title' 列
df.drop(columns='Title', inplace=True)
 
# 5. 验证缺失值是否已处理
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
X = df.drop('Survived', axis=1) 
y = df['Survived']               
# %%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # 固定随机种子保证可重复性
    stratify=y        # 保持类别分布一致（针对分类问题）
)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Testing set: {X_test.shape[0]} samples')
# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # 新增逻辑回归
 
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='rbf', probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}
# %%
# predict and evaluate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
 
def evaluate_model(model, X_test, y_test, model_name):
    """综合评估模型性能"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率（用于ROC曲线）
    
    print(f'=== {model_name} Performance ===')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred):.4f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\n' + '='*50 + '\n')
# 训练并评估所有模型
for name, model in models.items():
# 训练模型
    model.fit(X_train, y_train)
    
# 评估模型
    evaluate_model(model, X_test, y_test, name)
# %%
