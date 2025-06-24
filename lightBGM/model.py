import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna

# === IQR 기반 이상치 제거 함수 ===
def remove_outliers_iqr(df, columns, factor=1.5):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "call119_train.csv")

# === 데이터 로딩 및 전처리 ===
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.replace("call119_train.", "")

df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d')
df['year'] = df['tm'].dt.year
df['month'] = df['tm'].dt.month
df['day'] = df['tm'].dt.day
df['weekday'] = df['tm'].dt.weekday

# 정렬 및 lag 변수
df = df.sort_values(['address_gu', 'tm'])
df['lag_1'] = df.groupby('address_gu')['call_count'].shift(1)
df['lag_7'] = df.groupby('address_gu')['call_count'].shift(7)
df.dropna(inplace=True)

# Feature Engineering
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['is_rainy'] = (df['rn_day'] > 0).astype(int)
df['hm_range'] = df['hm_max'] - df['hm_min']
df['wind_range'] = df['ws_ins_max'] - df['ws_max']
df['season'] = df['month'] % 12 // 3 + 1

# Label Encoding
for col in ['address_city', 'address_gu', 'stn', 'season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# === 이상치 제거 ===
target_cols = [
    "ta_max", "ta_min", "hm_max", "hm_min", "rn_day",
    "ws_max", "ws_ins_max", "call_count"
]
df = remove_outliers_iqr(df, target_cols)

# === 지역별 Optuna 튜닝 + LightGBM 학습 ===
all_preds = []
all_targets = []
gu_list = df['address_gu'].unique()

for gu in gu_list:
    df_gu = df[df['address_gu'] == gu]
    train_df = df_gu[df_gu['year'] <= 2022]
    val_df = df_gu[df_gu['year'] == 2023]

    if len(val_df) == 0 or len(train_df) < 100:
        continue

    X_train = train_df.drop(columns=['call_count', 'tm', 'sub_address'])
    y_train = train_df['call_count']
    X_val = val_df.drop(columns=['call_count', 'tm', 'sub_address'])
    y_val = val_df['call_count']

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_model = LGBMRegressor(**study.best_params)
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_val)

    all_preds.extend(pred)
    all_targets.extend(y_val)

# 최종 RMSE
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
print(f"✅ 지역별 Optuna 튜닝 + 이상치 제거 + LightGBM 최종 RMSE: {rmse:.4f}")
