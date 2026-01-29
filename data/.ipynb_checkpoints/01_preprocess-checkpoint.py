# 1. 필수 라이브러리 임포트
import pandas as pd
import numpy as np
import joblib # 데이터 저장/로드용
import gc     # 메모리 청소용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2. 신용카드 거래 데이터 로드
df = pd.read_csv('creditcard.csv')

# 3. 패턴 추출 및 로그 변환
df['Hour'] = (df['Time'] // 3600) % 24
df['Log_Amount'] = np.log1p(df['Amount'])

# 4. 피처 및 타겟 분리
X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']

# 5. Train/Test 분리 (8:2, 층화 추출)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# [핵심] 메모리 절약: 원본 데이터프레임 삭제
del df
gc.collect()

# 6. 표준화 스케일링 (Standardization)
scaler = StandardScaler()

# 메모리 효율을 위해 float32로 변환하여 저장합니다. (용량 50% 절감)
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# 7. 파일 저장 ( joblib 사용 )
print("💾 데이터를 파일로 저장 중...")
joblib.dump(X_train_scaled, 'X_train_scaled.pkl')
joblib.dump(X_test_scaled, 'X_test_scaled.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(scaler, 'scaler.pkl') # 나중에 새로운 데이터 예측 시 필요

print("✅ 모든 데이터가 성공적으로 저장되었습니다.")
print(f"- Train Shape: {X_train_scaled.shape}")
print(f"- Test Shape: {X_test_scaled.shape}")

# 8. 최종 메모리 정리
del X_train, X_test, X_train_scaled, X_test_scaled
gc.collect()