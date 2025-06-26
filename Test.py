#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 23:42:36 2025

@author: jooyoungjin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

# === 1. 데이터 불러오기 및 전처리 ===
df = pd.read_csv("call119_train.csv")
df.columns = [col.replace("call119_train.", "") for col in df.columns]
df.columns = df.columns.str.strip()

df["tm"] = pd.to_datetime(df["tm"], format="%Y%m%d")
df.replace(-99.0, np.nan, inplace=True)
df.dropna(inplace=True)

# 파생 변수 생성
df["month"] = df["tm"].dt.month
df["day"] = df["tm"].dt.day
df["weekday"] = df["tm"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
df["humidity_range"] = df["hm_max"] - df["hm_min"]
df["is_heavy_rain"] = (df["rn_day"] >= 50).astype(int)
df["is_heatwave"] = (df["ta_max"] >= 33).astype(int)

# 원핫 인코딩 적용
df = pd.get_dummies(df, columns=["address_gu"])

# Feature 선택
feature_cols = [
    'ta_max', 'ta_min', 'ta_max_min',
    'hm_min', 'hm_max', 'humidity_range',
    'ws_max', 'ws_ins_max', 'rn_day',
    'is_heavy_rain', 'is_heatwave',
    'month', 'day', 'weekday', 'is_weekend'
] + [col for col in df.columns if col.startswith('address_gu_')]

X = df[feature_cols]
y = df['call_count']

# === 2. 데이터 분할 ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. 하이퍼파라미터 튜닝 ===
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'max_features': ['sqrt', 'log2', None]
}

gbr = GradientBoostingRegressor(random_state=42)
random_search = RandomizedSearchCV(
    gbr,
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# === 4. 예측 및 성능 평가 ===
y_pred = best_model.predict(X_val)
y_true = y_val

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("\n✅ 최적 하이퍼파라미터:", random_search.best_params_)
print(f"✅ 튜닝 후 GradientBoosting RMSE: {rmse:.4f}")

# === 5. 예측 vs 실제 산점도 시각화 ===
y_pred_int = np.round(y_pred)
y_true_int = np.round(y_true)

df_eval = pd.DataFrame({'true': y_true_int, 'pred': y_pred_int})

plt.figure(figsize=(6, 6))
plt.scatter(df_eval['true'], df_eval['pred'], alpha=0.4, color='blue')
plt.plot([0, 30], [0, 30], color='red', linestyle='--', label='정확 예측선 (y=x)')
plt.xlabel("실제 신고건수")
plt.ylabel("예측 신고건수")
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.title("예측 vs 실제 신고건수 (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 검증용 2024년 데이터 예측

#df_test = pd.read_csv("test_call119.csv", encoding='cp949')
# (동일 전처리 수행)
# ...
#df_test = df_test[X.columns]  # 학습과 동일한 feature 사용
#y_test_pred = best_model.predict(df_test)
#df_test["call_count"] = np.round(y_test_pred).astype(int)
#df_test.to_csv("submission.csv", index=False, encoding='cp949')

