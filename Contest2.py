import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# ----------------------------
# 1. 학습 데이터 불러오기 및 전처리
# ----------------------------
df = pd.read_csv("call119_train.csv")
df.columns = [col.replace("call119_train.", "") for col in df.columns]
df["tm"] = pd.to_datetime(df["tm"], format="%Y%m%d")
df.replace(-99.0, np.nan, inplace=True)
df.dropna(inplace=True)

df["weekday"] = df["tm"].dt.dayofweek
df["month"] = df["tm"].dt.month
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["ta_diff"] = df["ta_max"] - df["ta_min"]
df["hm_diff"] = df["hm_max"] - df["hm_min"]
df["is_heatwave"] = (df["ta_max"] >= 33).astype(int)
df["is_heavy_rain"] = (df["rn_day"] >= 50).astype(int)

df = df.sort_values(["sub_address", "tm"])
df["call_count_lag1"] = df.groupby("sub_address")["call_count"].shift(1).fillna(0)
df["call_count_sum3"] = df.groupby("sub_address")["call_count"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_sum7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)
df["call_count_mean7"] = df.groupby("sub_address")["call_count"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
df["rn_day_lag1"] = df.groupby("sub_address")["rn_day"].shift(1).fillna(0)
df["rn_day_sum3"] = df.groupby("sub_address")["rn_day"].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
df["rn_day_sum7"] = df.groupby("sub_address")["rn_day"].rolling(7, min_periods=1).sum().reset_index(0, drop=True)

df = pd.get_dummies(df, columns=["stn", "address_gu", "sub_address"])

features = [
    "ta_max", "ta_min", "ta_max_min", "hm_min", "hm_max", "ws_max", "ws_ins_max", "rn_day",
    "ta_diff", "hm_diff", "weekday", "month",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "is_heatwave", "is_heavy_rain",
    "call_count_lag1", "call_count_sum3", "call_count_sum7", "call_count_mean7",
    "rn_day_lag1", "rn_day_sum3", "rn_day_sum7"
] + [col for col in df.columns if col.startswith(("stn_", "address_gu_", "sub_address_"))]

X = df[features]
y = df["call_count"]

# ----------------------------
# 2. 모델 학습
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gbr = GradientBoostingRegressor(
    n_estimators=388,
    learning_rate=0.173,
    max_depth=3,
    min_samples_split=4,
    min_samples_leaf=3,
    subsample=0.842,
    max_features=None,
    random_state=42
)
gbr.fit(X_train, y_train)

# ----------------------------
# 3. 평가
# ----------------------------
y_pred = gbr.predict(X_test)
rmse_total = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📊 전체 RMSE: {rmse_total:.4f}")
rmse_high = np.sqrt(mean_squared_error(y_test[y_test >= 10], y_pred[y_test >= 10]))
print(f"🔴 고신고일 RMSE (y ≥ 10): {rmse_high:.4f}")

true_flag = (y_test >= 10).astype(int)
pred_flag = (y_pred >= 10).astype(int)
print("\n🧠 고신고일 경보 플래그 평가")
print(confusion_matrix(true_flag, pred_flag))
print(classification_report(true_flag, pred_flag, digits=4))

# ----------------------------
# 4. 시각화
# ----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(np.round(y_test), np.round(y_pred), alpha=0.4, color='blue')
plt.plot([0, 20], [0, 20], 'r--', label='정확 예측선 (y=x)')
plt.xlabel("실제 신고건수")
plt.ylabel("예측 신고건수")
plt.title("GradientBoosting 예측 vs 실제")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. 테스트셋 예측 및 제출 파일 저장 ===

# 1. 테스트셋 불러오기
df_test = pd.read_csv("call119_test.csv", encoding='cp949')
df_test.columns = df_test.columns.str.strip().str.lower()  # 소문자 처리

# 2. 전처리 동일하게 적용
df_test["tm"] = pd.to_datetime(df_test["tm"], format="%Y%m%d")
df_test["month"] = df_test["tm"].dt.month
df_test["day"] = df_test["tm"].dt.day
df_test["weekday"] = df_test["tm"].dt.weekday
df_test["is_weekend"] = df_test["weekday"].isin([5, 6]).astype(int)
df_test["humidity_range"] = df_test["hm_max"] - df_test["hm_min"]
df_test["is_heavy_rain"] = (df_test["rn_day"] >= 50).astype(int)
df_test["is_heatwave"] = (df_test["ta_max"] >= 33).astype(int)

df_test = pd.get_dummies(df_test, columns=["address_gu"])

# 누락된 컬럼 0으로 채우고 순서 맞추기
missing_cols = set(X.columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0
df_test = df_test[X.columns]

# 3. 예측 수행
y_test_pred = gbr.predict(df_test)
call_count_pred = np.round(y_test_pred).astype(int)

# 4. 제출 파일 생성 및 저장
submission = pd.DataFrame({
    "id": range(len(call_count_pred)),
    "call_count": call_count_pred
})
submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")
print("✅ submission.csv 저장 완료!")