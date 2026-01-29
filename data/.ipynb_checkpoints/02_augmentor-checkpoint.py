# 1. 필수 라이브러리 및 도구 임포트
import numpy as np
import joblib
import gc  # 가비지 컬렉터 (메모리 강제 비우기)
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# 2. 증강 목표 비율 설정
target_ratio = 0.2
device_cpu = 'float32' # 메모리 절약을 위해 float32 권장

# --- [공통 데이터 준비] ---
# 이전 셀에서 생성된 X_train_scaled, y_train을 사용합니다.
# 이미 메모리에 있다면 그대로 사용하고, 없다면 아래 주석을 풀어 로드하세요.
X_train_scaled = joblib.load('X_train_scaled.pkl')
y_train = joblib.load('y_train.pkl')

# --- [방법 A] Original: 증강하지 않은 데이터 저장 ---
print("💾 [1/4] Original 데이터 저장 중...")
X_train_org = X_train_scaled.astype(device_cpu)
y_train_org = y_train.values
joblib.dump(X_train_org, 'X_train_org.pkl')
joblib.dump(y_train_org, 'y_train_org.pkl')

# 메모리 비우기
del X_train_org, y_train_org
gc.collect()

# --- [방법 B] SMOTE: 증강 및 저장 ---
print("🚀 [2/4] SMOTE 증강 및 저장 중...")
smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

joblib.dump(X_train_smote.astype(device_cpu), 'X_train_smote.pkl')
joblib.dump(y_train_smote, 'y_train_smote.pkl')

del X_train_smote, y_train_smote
gc.collect()

# --- [방법 C] cGAN: 단순 생성 및 저장 ---
print("🚀 [3/4] cGAN 생성 및 저장 중...")
fraud_indices = np.where(y_train == 1)[0]
fraud_mean = X_train_scaled[fraud_indices].mean(axis=0)
fraud_std = X_train_scaled[fraud_indices].std(axis=0)

needed_cgan = int(len(X_train_scaled[y_train == 0]) * target_ratio) - len(fraud_indices)
fake_cgan = np.random.normal(fraud_mean, fraud_std * 0.25, size=(needed_cgan, X_train_scaled.shape[1]))

X_train_cgan = np.vstack([X_train_scaled, fake_cgan]).astype(device_cpu)
y_train_cgan = np.append(y_train.values, np.ones(needed_cgan))

joblib.dump(X_train_cgan, 'X_train_cgan.pkl')
joblib.dump(y_train_cgan, 'y_train_cgan.pkl')

del X_train_cgan, y_train_cgan, fake_cgan
gc.collect()

# --- [방법 D] K-cGAN: 군집 기반 생성 및 저장 ---
print("🚀 [4/4] K-cGAN 생성 및 저장 중...")
X_fraud_raw = X_train_scaled[fraud_indices]
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_fraud_raw)

needed_kcgan = int(len(X_train_scaled[y_train == 0]) * target_ratio) - len(fraud_indices)
gen_per_cluster = needed_kcgan // 10
gen_samples_kcgan = []

for i in range(10):
    cluster_subset = X_fraud_raw[clusters == i]
    fake_subset = np.random.normal(cluster_subset.mean(axis=0), 
                                   cluster_subset.std(axis=0) * 0.25, 
                                   size=(gen_per_cluster, X_train_scaled.shape[1]))
    gen_samples_kcgan.append(fake_subset)

X_train_kcgan = np.vstack([X_train_scaled, np.vstack(gen_samples_kcgan)]).astype(device_cpu)
y_train_kcgan = np.append(y_train.values, np.ones(len(np.vstack(gen_samples_kcgan))))

joblib.dump(X_train_kcgan, 'X_train_kcgan.pkl')
joblib.dump(y_train_kcgan, 'y_train_kcgan.pkl')

del X_train_kcgan, y_train_kcgan, gen_samples_kcgan, X_fraud_raw
gc.collect()

print("\n✅ 모든 증강 데이터셋이 개별 파일(.pkl)로 저장되었습니다.")