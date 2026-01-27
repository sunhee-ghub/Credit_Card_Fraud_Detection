import numpy as np
import pandas as pd
import joblib
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

# 1. 공통 데이터 로드 (테스트 데이터 및 스케일러)
print("🔍 공통 데이터를 로드 중...")
X_test_scaled = joblib.load('X_test_scaled.pkl')
y_test = joblib.load('y_test.pkl')

# 2. 실험할 데이터셋 목록 (조원분이 만든 파일 접미사들)
# 파일명이 X_train_org.pkl, X_train_smote.pkl 등임을 가정합니다.
data_variants = ['org', 'smote', 'cgan', 'kcgan']
results_rf = []

# 설정값
N_TREE = 200
CUSTOM_THRESHOLD = 0.38

print(f"\n[실험 시작] 랜덤포레스트 (Threshold={CUSTOM_THRESHOLD})")
print("="*60)

for variant in data_variants:
    print(f"🔄 [{variant.upper()}] 데이터셋 학습 및 평가 중...")
    
    try:
        # 데이터 로드
        X_tr = joblib.load(f'X_train_{variant}.pkl')
        y_tr = joblib.load(f'y_train_{variant}.pkl')
        
        # 모델 학습
        rf = RandomForestClassifier(
            n_estimators=N_TREE,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_tr, y_tr)
        
        # 예측 및 확률 계산
        y_probs = rf.predict_proba(X_test_scaled)[:, 1]
        y_pred_new = (y_probs >= CUSTOM_THRESHOLD).astype(int)
        
        # 성능 지표 계산
        results_rf.append({
            "Method": variant.upper(),
            "F1-Score": f1_score(y_test, y_pred_new),
            "Recall": recall_score(y_test, y_pred_new),
            "Precision": precision_score(y_test, y_pred_new),
            "ROC-AUC": roc_auc_score(y_test, y_probs)
        })
        
        # 메모리 정리 (매우 중요)
        del X_tr, y_tr, rf
        gc.collect()
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: X_train_{variant}.pkl")
    except Exception as e:
        print(f"❌ 에러 발생 ({variant}): {e}")

# 3. 최종 결과 리포트 출력
results_df = pd.DataFrame(results_rf)
pd.options.display.float_format = '{:.4f}'.format

print("\n" + "="*60)
print("🏆 최종 실험 결과 비교")
print("="*60)
if not results_df.empty:
    # F1-Score 기준으로 내림차순 정렬하여 출력
    print(results_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))