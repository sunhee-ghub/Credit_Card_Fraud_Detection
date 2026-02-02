import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, average_precision_score, precision_recall_curve)
import os

# 1. 경로 및 장치 설정
DATA_PATH = "./data_pipeline/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. AutoEncoder 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 10), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(),
            nn.Linear(20, input_dim) # StandardScaler 값 복원을 위해 Tanh 제거
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def evaluate_ae():
    # 평가 대상 데이터셋 명칭 및 파일명
    methods = [
        ("SMOTE", "train_smote.csv"),
        ("cGAN", "train_cgan.csv"),
        ("K-cGAN", "train_kcgan.csv")
    ]
    
    final_results = []

    for name, file_name in methods:
        print(f"\n🚀 {name} 데이터셋 분석 시작...")
        
        file_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ {file_name} 파일이 존재하지 않아 건너뜁니다.")
            continue
            
        # 데이터 로드
        df = pd.read_csv(file_path)
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        
        # 3. 8:2 분할 (층화 추출로 사기 비율 유지)
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. [중요] 학습은 80%의 데이터 중 '정상(0)'만 추출하여 진행
        X_train_normal = X_train_all[y_train_all == 0]
        
        # 모델 및 학습 설정
        model = AutoEncoder(X_train_normal.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_normal).to(device)), 
                            batch_size=512, shuffle=True)
        
        # 5. 모델 학습 (정상 패턴 학습)
        model.train()
        for epoch in range(50):
            for [batch] in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
        
        # 6. 평가 (재구축 오차 MSE 계산)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            reconstructed = model(X_test_tensor)
            mse_scores = torch.mean((X_test_tensor - reconstructed)**2, dim=1).cpu().numpy()
        
        # 7. 최적 임계치(Threshold) 탐색 및 지표 계산
        precisions, recalls, thresholds = precision_recall_curve(y_test, mse_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        
        # thresholds 길이 맞춤형 인덱싱 (IndexError 방지)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        
        # 최종 예측값 (임계치 적용)
        preds = (mse_scores > best_threshold).astype(int)

        # 결과 딕셔너리에 추가
        final_results.append({
            "Method": name,
            "Threshold": round(float(best_threshold), 6),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "ROC-AUC": round(roc_auc_score(y_test, mse_scores), 4),
            "AUPRC": round(average_precision_score(y_test, mse_scores), 4)
        })
        print(f"✅ {name} 완료 (Recall: {final_results[-1]['Recall']})")
        
        results_df = pd.DataFrame(final_results)

if __name__ == "__main__":
    evaluate_ae()