import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 경로 설정
DATA_PATH = "./data_pipeline/"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def preprocess_raw_data():
    print("Step 0: Preprocessing Raw Data...")
    # 파일이 같은 경로에 있는지 확인하세요
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: 'creditcard.csv' 파일이 없습니다. 경로를 확인하세요.")
        return

    # 1. 'Time' 컬럼 변환 (초 -> 시간)
    df['Time'] = (df['Time'] // 3600) % 24
    
    # 2. 피처와 레이블 분리
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 3. 전체 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 전처리 완료된 데이터 저장
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed['Class'] = y.values
    
    output_file = os.path.join(DATA_PATH, "base_preprocessed.csv")
    df_preprocessed.to_csv(output_file, index=False)
    print(f"✅ Success! Saved to {output_file}")

if __name__ == "__main__":
    preprocess_raw_data()