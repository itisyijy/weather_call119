import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 학습 데이터 로딩 및 전처리
df_train = pd.read_csv("call119_train.csv")
df_train.columns = [col.replace("call119_train.", "") for col in df_train.columns]
df_train.columns = df_train.columns.str.strip()

# 날짜 파싱 및 파생 변수
df_train['tm'] = pd.to_datetime(df_train['tm'], format='%Y%m%d')
df_train['month'] = df_train['tm'].dt.month
df_train['day'] = df_train['tm'].dt.day
df_train['weekday'] = df_train['tm'].dt.weekday
df_train['is_weekend'] = df_train['weekday'].isin([5, 6]).astype(int)

df_train['humidity_range'] = df_train['hm_max'] - df_train['hm_min']
df_train['is_heavy_rain'] = (df_train['rn_day'] >= 50).astype(int)
df_train['is_heatwave'] = (df_train['ta_max'] >= 33).astype(int)

# 원핫 인코딩
df_train = pd.get_dummies(df_train, columns=['address_gu'])

# 학습용 feature 정의
feature_cols = [
    'ta_max', 'ta_min', 'ta_max_min',
    'hm_min', 'hm_max', 'humidity_range',
    'ws_max', 'ws_ins_max', 'rn_day',
    'is_heavy_rain', 'is_heatwave',
    'month', 'day', 'weekday', 'is_weekend'
] + [col for col in df_train.columns if col.startswith('address_gu_')]

X = df_train[feature_cols]
y = df_train['call_count']

# 모델 학습
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 2. 테스트 데이터 로딩 및 동일 전처리
df_test = pd.read_csv("test_call119.csv", encoding='cp949')
df_test.columns = df_test.columns.str.strip()

# 첫 번째 열이 'tm'이 아닐 경우 처리
if 'tm' not in df_test.columns:
    df_test.rename(columns={df_test.columns[0]: 'tm'}, inplace=True)

df_test['tm'] = pd.to_datetime(df_test['tm'], format='%Y%m%d')
df_test['month'] = df_test['tm'].dt.month
df_test['day'] = df_test['tm'].dt.day
df_test['weekday'] = df_test['tm'].dt.weekday
df_test['is_weekend'] = df_test['weekday'].isin([5, 6]).astype(int)

df_test['humidity_range'] = df_test['hm_max'] - df_test['hm_min']
df_test['is_heavy_rain'] = (df_test['rn_day'] >= 50).astype(int)
df_test['is_heatwave'] = (df_test['ta_max'] >= 33).astype(int)

# 원핫 인코딩 정렬
df_test = pd.get_dummies(df_test, columns=['address_gu'])
for col in X.columns:
    if col not in df_test.columns:
        df_test[col] = 0
df_test = df_test[X.columns]

# 예측
y_pred = model.predict(df_test)

# 결과 저장
df_submit = pd.read_csv("test_call119.csv", encoding='cp949')
df_submit['call_count'] = np.round(y_pred).astype(int)
df_submit.to_csv("submission_gradientboosting.csv", index=False, encoding='cp949')
print("✅ 제출 파일 저장 완료: submission_gradientboosting.csv")