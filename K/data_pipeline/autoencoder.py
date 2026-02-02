import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, average_precision_score)
import os
import gc

DATA_PATH = "./data_pipeline/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AutoEncoder 모델 구조
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def evaluate_ae():
    data_files = [("SMOTE", "train_smote.csv"), ("cGAN", "train_cgan.csv"), ("K-cGAN", "train_kcgan.csv")]
    final_results = []

    for name, file_name in data_files:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path): continue

        print(f"🚀 [{name}] AE 학습 및 평가 중...")
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        # 8:2 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # [AE 핵심] 학습은 정상(Class 0) 데이터로만 수행
        X_train_normal = X_train[y_train == 0]
        
        model = AutoEncoder(X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_normal).to(device)), batch_size=512, shuffle=True)
        
        model.train()
        for epoch in range(30):
            for [batch] in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward(); optimizer.step()
        
        # 평가 (재구축 오차 계산)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            reconstructed = model(X_test_tensor)
            mse_scores = torch.mean((X_test_tensor - reconstructed)**2, dim=1).cpu().numpy()
        
        # 임계치 결정 (오차 상위 5%를 사기로 간주)
        threshold = np.percentile(mse_scores, 95)
        preds = (mse_scores > threshold).astype(int)

        final_results.append({
            "Method": name,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, mse_scores),
            "AUPRC": average_precision_score(y_test, mse_scores)
        })
        del df, X_train, X_test; gc.collect()

    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 AutoEncoder 기반 증강 기법별 성능 비교 (8:2 Split)")
    print("="*70)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_ae()