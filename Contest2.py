#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 23:10:37 2025

@author: jooyoungjin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 불러오기
df = pd.read_csv("call119_train.csv")

# 결측치 처리
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].replace(-99.0, np.nan)
df.dropna(inplace=True)

# X, y 정의 (기본 변수만 사용)
X = df[[
    "call119_train.ta_max", "call119_train.ta_min", "call119_train.ta_max_min",
    "call119_train.hm_min", "call119_train.hm_max",
    "call119_train.ws_max", "call119_train.ws_ins_max",
    "call119_train.rn_day"
]]
y = df["call119_train.call_count"]

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("call119_train.csv")
df.columns = [col.replace("call119_train.", "") for col in df.columns]
df["tm"] = pd.to_datetime(df["tm"], format="%Y%m%d")
df.replace(-99.0, np.nan, inplace=True)
df.dropna(inplace=True)

# 2. 시간 파생 변수
df["weekday"] = df["tm"].dt.dayofweek
df["month"] = df["tm"].dt.month
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 3. 공휴일 플래그
holidays = pd.to_datetime([
    '2020-01-01','2020-01-25','2020-01-26','2020-01-27','2020-03-01','2020-05-05','2020-05-08',
    '2020-06-06','2020-08-15','2020-08-17','2020-09-30','2020-10-01','2020-10-02','2020-10-03','2020-10-09','2020-12-25',
    '2021-01-01','2021-02-11','2021-02-12','2021-02-13','2021-03-01','2021-05-05','2021-05-19',
    '2021-06-06','2021-08-15','2021-08-16','2021-09-20','2021-09-21','2021-09-22','2021-10-03','2021-10-04','2021-10-09','2021-10-11','2021-12-25',
    '2022-01-01','2022-01-31','2022-02-01','2022-02-02','2022-03-01','2022-05-05','2022-05-08',
    '2022-06-06','2022-08-15','2022-09-09','2022-09-10','2022-09-11','2022-10-03','2022-10-09','2022-12-25',
    '2023-01-01','2023-01-21','2023-01-22','2023-01-23','2023-01-24','2023-03-01','2023-05-05','2023-05-27',
    '2023-06-06','2023-08-15','2023-09-28','2023-09-29','2023-09-30','2023-10-02','2023-10-03','2023-10-09','2023-12-25'
])
df["is_holiday"] = df["tm"].isin(holidays).astype(int)

# 4. 파생 변수
df["ta_diff"] = df["ta_max"] - df["ta_min"]
df["hm_diff"] = df["hm_max"] - df["hm_min"]
df["is_heatwave"] = (df["ta_max"] >= 33).astype(int)
df["is_heavy_rain"] = (df["rn_day"] >= 50).astype(int)

# 5. 시계열 변수
df = df.sort_values(["sub_address", "tm"])
df["call_count_lag1"] = df.groupby("sub_address")["call_count"].shift(1).fillna(0)
df["call_count_sum3"] = df.groupby("sub_address")["call_count"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_sum7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_mean7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
df["rn_day_lag1"] = df.groupby("sub_address")["rn_day"].shift(1).fillna(0)
df["rn_day_sum3"] = df.groupby("sub_address")["rn_day"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["rn_day_sum7"] = df.groupby("sub_address")["rn_day"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)

# 6. 지역 인코딩
df = pd.get_dummies(df, columns=["stn", "address_gu", "sub_address"])

# 7. 학습 및 평가
features = [
    "ta_max", "ta_min", "ta_max_min", "hm_min", "hm_max", "ws_max", "ws_ins_max", "rn_day",
    "ta_diff", "hm_diff", "weekday", "month",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "is_holiday", "is_heatwave", "is_heavy_rain",
    "call_count_lag1", "call_count_sum3", "call_count_sum7", "call_count_mean7",
    "rn_day_lag1", "rn_day_sum3", "rn_day_sum7"
] + [col for col in df.columns if col.startswith(("stn_", "address_gu_", "sub_address_"))]

X = df[features]
y = df["call_count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# 앙상블 평균
ensemble_pred = (rf_pred + gb_pred) / 2

# 성능 평가
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
rmse_gb = np.sqrt(mean_squared_error(y_test, gb_pred))
rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"✅ RandomForest RMSE: {rmse_rf:.4f}")
print(f"✅ GradientBoosting RMSE: {rmse_gb:.4f}")
print(f"✅ Ensemble RMSE: {rmse_ens:.4f}")




