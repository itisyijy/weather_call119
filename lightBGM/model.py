import pandas as pd
import os
import optuna
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "call119_train.csv")

# === 데이터 로딩 ===
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.replace("call119_train.", "")
df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d')
df['year'] = df['tm'].dt.year
df['month'] = df['tm'].dt.month
df['day'] = df['tm'].dt.day
df['weekday'] = df['tm'].dt.weekday

# === Feature Engineering ===
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['is_rainy'] = (df['rn_day'] > 0).astype(int)
df['hm_range'] = df['hm_max'] - df['hm_min']
df['wind_range'] = df['ws_ins_max'] - df['ws_max']
df['season'] = df['month'] % 12 // 3 + 1

# === Label Encoding ===
for col in ['address_city', 'address_gu', 'stn', 'season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# === Train / Validation 분할 ===
train_df = df[df['year'] <= 2022]
val_df = df[df['year'] == 2023]

X_train = train_df.drop(columns=['call_count', 'tm', 'sub_address'])
y_train = train_df['call_count']
X_val = val_df.drop(columns=['call_count', 'tm', 'sub_address'])
y_val = val_df['call_count']

# === Optuna 목적함수 정의 ===
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 700),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# === Optuna 실행 ===
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# === 결과 출력 ===
print("✅ Best RMSE:", study.best_value)
print("✅ Best Params:", study.best_params)

# === 최적 모델 학습 및 평가 ===
best_model = LGBMRegressor(**study.best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)

final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"✅ 최적화된 LightGBM RMSE: {final_rmse:.4f}")
