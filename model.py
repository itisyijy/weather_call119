import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 1. 학습 데이터 로딩 및 전처리
df_train = pd.read_csv("call119_train.csv")
df_train.columns = [col.replace("call119_train.", "") for col in df_train.columns]
df_train.columns = df_train.columns.str.strip()

# 날짜 파싱
df_train["tm"] = pd.to_datetime(df_train["tm"], format="%Y%m%d")

# -99 결측치 처리
df_train.replace(-99.0, np.nan, inplace=True)
df_train.dropna(inplace=True)

# 날짜 기반 파생 변수
df_train["weekday"] = df_train["tm"].dt.weekday
df_train["month"] = df_train["tm"].dt.month
df_train["weekday_sin"] = np.sin(2 * np.pi * df_train["weekday"] / 7)
df_train["weekday_cos"] = np.cos(2 * np.pi * df_train["weekday"] / 7)
df_train["month_sin"] = np.sin(2 * np.pi * df_train["month"] / 12)
df_train["month_cos"] = np.cos(2 * np.pi * df_train["month"] / 12)

# 공휴일 여부
holidays = pd.to_datetime([
    '2020-01-01','2020-01-25','2020-01-26','2020-01-27','2020-03-01','2020-05-05','2020-05-08','2020-06-06','2020-08-15','2020-08-17',
    '2020-09-30','2020-10-01','2020-10-02','2020-10-03','2020-10-09','2020-12-25','2021-01-01','2021-02-11','2021-02-12','2021-02-13',
    '2021-03-01','2021-05-05','2021-05-19','2021-06-06','2021-08-15','2021-08-16','2021-09-20','2021-09-21','2021-09-22','2021-10-03',
    '2021-10-04','2021-10-09','2021-10-11','2021-12-25','2022-01-01','2022-01-31','2022-02-01','2022-02-02','2022-03-01','2022-05-05',
    '2022-05-08','2022-06-06','2022-08-15','2022-09-09','2022-09-10','2022-09-11','2022-10-03','2022-10-09','2022-12-25','2023-01-01',
    '2023-01-21','2023-01-22','2023-01-23','2023-01-24','2023-03-01','2023-05-05','2023-05-27','2023-06-06','2023-08-15','2023-09-28',
    '2023-09-29','2023-09-30','2023-10-02','2023-10-03','2023-10-09','2023-12-25','2024-01-01','2024-02-09','2024-02-10','2024-02-11',
    '2024-02-12','2024-03-01','2024-05-05','2024-05-15','2024-06-06','2024-08-15','2024-09-16','2024-09-17','2024-09-18','2024-10-03',
    '2024-10-09','2024-12-25'
])
df_train["is_holiday"] = df_train["tm"].isin(holidays).astype(int)

# 기상 파생 변수
df_train["ta_diff"] = df_train["ta_max"] - df_train["ta_min"]
df_train["hm_diff"] = df_train["hm_max"] - df_train["hm_min"]
df_train["is_heatwave"] = (df_train["ta_max"] >= 33).astype(int)
df_train["is_heavy_rain"] = (df_train["rn_day"] >= 50).astype(int)

# 인코딩
df_train = pd.get_dummies(df_train, columns=["stn", "address_gu", "sub_address"])

# 특성 정의
features = [
    "ta_max", "ta_min", "ta_max_min", "hm_min", "hm_max", "ws_max", "ws_ins_max", "rn_day",
    "ta_diff", "hm_diff", "weekday", "month", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "is_holiday", "is_heatwave", "is_heavy_rain"
] + [col for col in df_train.columns if col.startswith(("stn_", "address_gu_", "sub_address_"))]

X = df_train[features]
y = np.log1p(df_train["call_count"])  # log1p 변환된 타겟 사용
# ---------------------------

# 2. 모델 학습 및 검증

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE", verbose=100)
model.fit(X_train, y_train)

# 검증 RMSE (복원 후 계산)
val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f" validation RMSE: {rmse:.4f}")
# ---------------------------

# 3. 테스트셋 예측 및 저장

df_test = pd.read_csv("test_call119.csv", encoding="cp949")
df_test.columns = df_test.columns.str.strip().str.lower()
df_test["tm"] = pd.to_datetime(df_test["tm"], format="%Y%m%d")

df_test["month"] = df_test["tm"].dt.month
df_test["weekday"] = df_test["tm"].dt.weekday
df_test["weekday_sin"] = np.sin(2 * np.pi * df_test["weekday"] / 7)
df_test["weekday_cos"] = np.cos(2 * np.pi * df_test["weekday"] / 7)
df_test["month_sin"] = np.sin(2 * np.pi * df_test["month"] / 12)
df_test["month_cos"] = np.cos(2 * np.pi * df_test["month"] / 12)
df_test["is_holiday"] = df_test["tm"].isin(holidays).astype(int)
df_test["ta_diff"] = df_test["ta_max"] - df_test["ta_min"]
df_test["hm_diff"] = df_test["hm_max"] - df_test["hm_min"]
df_test["is_heatwave"] = (df_test["ta_max"] >= 33).astype(int)
df_test["is_heavy_rain"] = (df_test["rn_day"] >= 50).astype(int)

# 인코딩
df_test = pd.get_dummies(df_test, columns=["stn", "address_gu", "sub_address"])
for col in X.columns:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test[X.columns]

# 예측 및 복원
pred_log = model.predict(df_test)
pred = np.expm1(pred_log)
pred = np.round(pred).astype(int)

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------

# 4. 예측 및 저장

# 1. 검증셋 예측
val_pred_log = model.predict(X_val)             # 로그 스케일 예측
val_pred = np.expm1(val_pred_log)               # 역변환 (log1p → 원래 scale)
y_val_actual = np.expm1(y_val)                  # y_val도 역변환

# 2. 시각화
plt.figure(figsize=(8, 8))
plt.scatter(y_val_actual, val_pred, alpha=0.4, color='blue')
plt.plot([0, 20], [0, 20], 'r--', label='완벽 예측 (y=x)')
plt.title("실제 vs 모델 예측값 (Validation Set)")
plt.xlabel("실제 call_count")
plt.ylabel("예측 call_count")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 저장
df_submit = pd.read_csv("test_call119.csv", encoding="cp949")
df_submit["call_count"] = pred
df_submit.to_csv("submission_catboost_log1p.csv", index=False, encoding="cp949")
print("submission_catboost_log1p.csv")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# 로그 복원
y_val_true = np.expm1(y_val)
y_val_pred = np.expm1(val_pred_log)

# 고신고일 필터
high_threshold = 5
mask_high = y_val_true >= high_threshold

# 고신고일 대상
y_true_high = y_val_true[mask_high]
y_pred_high = y_val_pred[mask_high]

# 성능 지표
rmse_total = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
mae_total = mean_absolute_error(y_val_true, y_val_pred)

rmse_high = np.sqrt(mean_squared_error(y_true_high, y_pred_high))
mae_high = mean_absolute_error(y_true_high, y_pred_high)

# 시각화
plt.figure(figsize=(14, 6))

# 전체 예측
plt.subplot(1, 2, 1)
plt.scatter(y_val_true, y_val_pred, alpha=0.4, label="All samples")
plt.plot([0, max(y_val_true)], [0, max(y_val_true)], 'r--', label="y = x")
plt.xlabel("Actual call count")
plt.ylabel("Predicted call count")
plt.title(f"All data prediction\nRMSE={rmse_total:.2f}, MAE={mae_total:.2f}")
plt.legend()
plt.grid(True)

# 고신고일 예측
plt.subplot(1, 2, 2)
plt.scatter(y_true_high, y_pred_high, alpha=0.5, color='orange', label="High-call samples")
plt.plot([0, max(y_true_high)], [0, max(y_true_high)], 'r--', label="y = x")
plt.xlabel("Actual call count")
plt.ylabel("Predicted call count")
plt.title(f"High-call prediction (≥ {high_threshold})\nRMSE={rmse_high:.2f}, MAE={mae_high:.2f}")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 조건 필터링
predicted_high_mask = y_val_pred >= 5
num_predicted_high = np.sum(predicted_high_mask)
total_samples = len(y_val_pred)
ratio_predicted_high = num_predicted_high / total_samples
num_predicted_low = total_samples - num_predicted_high

# 파이 차트로 비율 표현 (선택적)
plt.figure(figsize=(5, 5))
plt.pie(
    [num_predicted_low, num_predicted_high],
    labels=["Predicted < 5", "Predicted ≥ 5"],
    autopct="%1.1f%%",
    colors=["skyblue", "orange"],
    startangle=140
)
plt.title("Proportion of Predicted High-call Days (≥ 5)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 복원된 실제값과 예측값
y_true = np.expm1(y_val)
y_pred = y_val_pred  # 이미 복원되어 있음

# 실제 고신고일 (call_count ≥ 5)
true_high = np.sum(y_true >= 5)
true_low = len(y_true) - true_high

# 예측 고신고일
pred_high = np.sum(y_pred >= 5)
pred_low = len(y_pred) - pred_high

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 실제값 기준 파이차트
axes[0].pie(
    [true_low, true_high],
    labels=["True < 5", "True ≥ 5"],
    autopct="%1.1f%%",
    colors=["skyblue", "orange"],
    startangle=140
)
axes[0].set_title("Actual Call Count ≥ 5")

# 예측값 기준 파이차트
axes[1].pie(
    [pred_low, pred_high],
    labels=["Predicted < 5", "Predicted ≥ 5"],
    autopct="%1.1f%%",
    colors=["skyblue", "orange"],
    startangle=140
)
axes[1].set_title("Predicted Call Count ≥ 5")

plt.suptitle("Proportion of High-call Days (call_count ≥ 5)", fontsize=14)
plt.tight_layout()
plt.show()
