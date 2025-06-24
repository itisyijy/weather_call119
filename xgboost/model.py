import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "call119_train.csv")
                               
# Load CSV file
df = pd.read_csv(CSV_PATH)

# Renaming Columns
df.columns = df.columns.str.replace("call119_train.", "")

# Check Columns
print(df.columns)

# Data Parsing
df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d')
df['year'] = df['tm'].dt.year
df['month'] = df['tm'].dt.month
df['day'] = df['tm'].dt.day
df['weekday'] = df['tm'].dt.weekday

# Feature Engineering (정리본)
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)            # 주말 여부
df['is_rainy'] = (df['rn_day'] > 0).astype(int)                      # 비가 온 날 여부
df['hm_range'] = df['hm_max'] - df['hm_min']                         # 습도 범위
df['wind_range'] = df['ws_ins_max'] - df['ws_max']                   # 돌풍 - 최대풍
df['season'] = df['month'] % 12 // 3 + 1                             # 계절 (1~4)


# Categorical Encoding
import optuna
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Label Encoding (추가된 season 포함)
for col in ['address_city', 'address_gu', 'stn', 'season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# Data Partitioning:
# 2020-2022 for Training
# 2023 for Validation
train_df = df[df['year'] <= 2022]
val_df = df[df['year'] == 2023]

X_train = train_df.drop(columns=['call_count', 'tm', 'sub_address'])
y_train = train_df['call_count']
X_val = val_df.drop(columns=['call_count', 'tm', 'sub_address'])
y_val = val_df['call_count']

# Optuna 목적 함수 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# 튜닝 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("✅ Best RMSE:", study.best_value)
print("✅ Best Params:", study.best_params)

best_params = study.best_params
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ 최적화된 XGBoost RMSE: {rmse:.4f}")
