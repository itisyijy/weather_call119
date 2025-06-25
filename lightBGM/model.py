import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import early_stopping

# Windows 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# === 현재 스크립트 위치로 작업 디렉토리 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === 데이터 로딩
def load_and_clean(filename):
    df = pd.read_csv(filename)
    df.columns = [col.lower().replace("call119_train.", "").replace("test_call119.", "") for col in df.columns]
    df['tm'] = pd.to_datetime(df['tm'])
    return df

train_df = load_and_clean("call119_train.csv")
test_df = load_and_clean("test_call119.csv")

# === 날짜 파생 변수
def base_preprocess(df):
    df['year'] = df['tm'].dt.year
    df['month'] = df['tm'].dt.month
    df['day'] = df['tm'].dt.day
    df['weekday'] = df['tm'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df = df.drop(columns=['tm', 'sub_address'])
    return df

# === 통합 처리
train_df['is_train'] = 1
test_df['is_train'] = 0
full_df = pd.concat([train_df, test_df], axis=0)
full_df = base_preprocess(full_df)
full_df = pd.get_dummies(full_df, columns=['address_city', 'address_gu', 'stn'], drop_first=True)
train_df = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
test_df = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])

# === 피크일 정의 (5건 이상)
TARGET = 'call_count'
train_df['is_peak'] = (train_df[TARGET] >= 5).astype(int)

# === Feature / Target 분리
X = train_df.drop(columns=[TARGET, 'is_peak'])
y_reg = np.log1p(np.clip(train_df[TARGET].values, 0, 15))
y_clf = train_df['is_peak'].values
X_test = test_df[X.columns]

# === 학습/검증 분할
n = len(X)
train_size = int(n * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train_reg, y_val_reg = y_reg[:train_size], y_reg[train_size:]
y_train_clf, y_val_clf = y_clf[:train_size], y_clf[train_size:]

# === 회귀 모델 (RMSE 출력)
reg_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_val)
rmse = np.sqrt(np.mean((np.expm1(y_val_reg) - np.expm1(y_pred_reg)) ** 2))
print(f"✅ 로그 변환 기반 RMSE: {rmse:.4f}")

# === 분류 모델 (threshold 조정 포함)
clf_model = lgb.LGBMClassifier(n_estimators=300, class_weight='balanced')
clf_model.fit(X_train, y_train_clf)
threshold = 0.7
y_proba_val = clf_model.predict_proba(X_val)[:, 1]
y_pred_clf = (y_proba_val >= threshold).astype(int)
print(f"✅ 피크일 분류 정확도 (threshold={threshold}): {np.mean(y_pred_clf == y_val_clf):.3f}")

# === 테스트셋 예측 (정수 후처리)
test_pred_reg = np.expm1(reg_model.predict(X_test))
test_proba_clf = clf_model.predict_proba(X_test)[:, 1]
test_pred_clf = (test_proba_clf >= threshold).astype(int)
adjusted_pred = test_pred_reg.copy()
adjusted_pred[test_pred_clf == 1] += 2
adjusted_pred = np.clip(np.round(adjusted_pred), 0, 15).astype(int)  # ✅ 정수화

submission = pd.DataFrame({
    'call_prediction': adjusted_pred,
    'peak_prediction': test_pred_clf
})
submission.to_csv("prediction.csv", index=False)
print("📁 결과 저장 완료: prediction.csv")

# === Optuna 튜닝 (회귀)
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'num_leaves': trial.suggest_int("num_leaves", 20, 150),
        'max_depth': trial.suggest_int("max_depth", 3, 15),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 2.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 2.0),
    }
    model = lgb.LGBMRegressor(**params, n_estimators=500)
    model.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)],
              eval_metric='rmse', callbacks=[early_stopping(30)])
    preds = model.predict(X_val)
    score = np.sqrt(np.mean((np.expm1(y_val_reg) - np.expm1(preds)) ** 2))
    print(f"[Trial Done] RMSE: {score:.4f}")
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("✅ Best RMSE:", study.best_value)
print("✅ Best params:", study.best_params)

# === 최적 회귀 모델 재학습
best_model = lgb.LGBMRegressor(**study.best_params, n_estimators=500)
best_model.fit(X_train, y_train_reg)
final_preds = best_model.predict(X_val)
final_preds_int = np.clip(np.round(np.expm1(final_preds)), 0, 15).astype(int)
true_val = np.expm1(y_val_reg)
rmse_final = np.sqrt(np.mean((true_val - final_preds_int) ** 2))
print(f"🔢 정수 기반 최적 RMSE: {rmse_final:.4f}")

# === 테스트셋 최종 예측
test_pred_reg = np.expm1(best_model.predict(X_test, num_iteration=best_model.best_iteration_))
adjusted_pred = test_pred_reg.copy()
adjusted_pred[test_pred_clf == 1] += 2
adjusted_pred = np.clip(np.round(adjusted_pred), 0, 15).astype(int)

submission = pd.DataFrame({
    'call_prediction': adjusted_pred,
    'peak_prediction': test_pred_clf
})
submission.to_csv("prediction_optuna_int.csv", index=False)
print("📁 Optuna 기반 정수 예측 결과 저장 완료: prediction_optuna_int.csv")

# === 피크일 분류 성능 평가
print("📊 피크일 분류 평가 지표:")
print(classification_report(y_val_clf, y_pred_clf, digits=3))
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_val_clf, y_pred_clf))

# === 예측 vs 실제 산점도
df_eval = pd.DataFrame({'true': true_val, 'pred': final_preds_int})
plt.figure(figsize=(6, 6))
plt.scatter(df_eval['true'], df_eval['pred'], alpha=0.4)
plt.plot([0, 15], [0, 15], color='red', linestyle='--', label='정확 예측선')
plt.xlabel("실제 신고건수")
plt.ylabel("예측 신고건수 (정수)")
plt.title("예측 vs 실제 신고건수 (Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Top-N 피크일 평가
def evaluate_top_n_accuracy(df, top_n=100):
    top_true = df.sort_values('true', ascending=False).head(top_n).index
    top_pred = df.sort_values('pred', ascending=False).head(top_n).index
    hit = len(set(top_true) & set(top_pred))
    print(f"🎯 Top-{top_n} 피크일 적중률: {hit}/{top_n} ({hit/top_n:.2%})")
    return hit / top_n

evaluate_top_n_accuracy(df_eval)
