import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# ---------------------------
# 데이터 로딩 및 전처리
# ---------------------------
print("🔹 데이터 불러오기 및 전처리 시작")
df = pd.read_csv("call119_train.csv")
df.columns = [col.replace("call119_train.", "") for col in df.columns]
df["tm"] = pd.to_datetime(df["tm"], format="%Y%m%d")
df.replace(-99.0, np.nan, inplace=True)
df.dropna(inplace=True)

# 시간 파생 변수
df["weekday"] = df["tm"].dt.dayofweek
df["month"] = df["tm"].dt.month
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 공휴일 플래그
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

# 파생 변수
df["ta_diff"] = df["ta_max"] - df["ta_min"]
df["hm_diff"] = df["hm_max"] - df["hm_min"]
df["is_heatwave"] = (df["ta_max"] >= 33).astype(int)
df["is_heavy_rain"] = (df["rn_day"] >= 50).astype(int)

# 시계열 변수
df = df.sort_values(["sub_address", "tm"])
df["call_count_lag1"] = df.groupby("sub_address")["call_count"].shift(1).fillna(0)
df["call_count_sum3"] = df.groupby("sub_address")["call_count"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_sum7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_mean7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
df["rn_day_lag1"] = df.groupby("sub_address")["rn_day"].shift(1).fillna(0)
df["rn_day_sum3"] = df.groupby("sub_address")["rn_day"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["rn_day_sum7"] = df.groupby("sub_address")["rn_day"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)

# 지역 인코딩
df = pd.get_dummies(df, columns=["stn", "address_gu", "sub_address"])

# Feature 정의
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

# ---------------------------
# 데이터 분할
# ---------------------------
print("✅ 전처리 완료, 데이터 분할 중")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 최적 GBM 하이퍼파라미터 적용
# ---------------------------
best_params = {
    'n_estimators': 165,
    'max_depth': 5,
    'learning_rate': 0.07057987286456649,
    'subsample': 0.7808872460773514,
    'random_state': 42
}

print("🔹 GradientBoosting 학습 시작")
start = time.time()
best_model = GradientBoostingRegressor(**best_params)
best_model.fit(X_train, y_train)
gb_pred = best_model.predict(X_test)
print(f"✅ GBM 학습 완료 ({time.time() - start:.2f}초)")

# Random Forest
print("🔹 RandomForest 학습 시작")
start = time.time()
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"✅ RF 학습 완료 ({time.time() - start:.2f}초)")

# 앙상블 예측
ensemble_pred = (rf_pred + gb_pred) / 2

# ---------------------------
# 정수 반올림 처리
# ---------------------------
gb_pred = np.round(gb_pred)
rf_pred = np.round(rf_pred)
ensemble_pred = np.round(ensemble_pred)

# ---------------------------
# 성능 평가 (정수 예측 기준 RMSE)
# ---------------------------
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
rmse_gb = np.sqrt(mean_squared_error(y_test, gb_pred))
rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_pred))

print(f"✅ [RMSE - 정수 예측 기준]")
print(f"   RandomForest:         {rmse_rf:.4f}")
print(f"   GradientBoosting:     {rmse_gb:.4f}")
print(f"   Ensemble (Avg):       {rmse_ens:.4f}")

# ---------------------------
# 산점도 시각화 (정수 예측 vs 실제)
# ---------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, gb_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Call Count")
plt.ylabel("Predicted (Rounded) Call Count")
plt.title("📈 GBM Prediction vs Actual (Rounded)")
plt.grid(True)
plt.tight_layout()
plt.show()
