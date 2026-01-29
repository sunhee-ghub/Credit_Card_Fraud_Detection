import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE

# 1. 경로 설정
DATA_PATH = "./data_pipeline/"
os.makedirs(DATA_PATH, exist_ok=True)

def run_smote_augmentation_only():
    print("🚀 Step 1: Loading preprocessed data...")
    input_file = f"{DATA_PATH}base_preprocessed.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} 파일이 없습니다.")
        return

    df = pd.read_csv(input_file)

    # 2. X, y 분리
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. SMOTE 1:1 증강 (전체 데이터를 대상으로 실시)
    print("📊 Applying SMOTE (1:1 Ratio for entire dataset)...")
    # k_neighbors 등 기본 설정 유지, sampling_strategy=1.0으로 1:1 맞춤
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # 4. 증강된 데이터를 하나의 데이터프레임으로 합침
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                             pd.Series(y_resampled, name='Class')], axis=1)
    
    print(f"✅ Augmentation Complete!")
    print(f"Original samples: {len(df)}")
    print(f"Total samples after SMOTE (1:1): {len(df_resampled)}")
    print(f"Class Distribution:\n{df_resampled['Class'].value_counts()}")

    # 5. 최종 증강 파일 저장 (Test 분리 없이 저장)
    output_file = f"{DATA_PATH}train_smote.csv"
    df_resampled.to_csv(output_file, index=False)
    
    print(f"💾 Saved: {output_file}")

if __name__ == "__main__":
    run_smote_augmentation_only()