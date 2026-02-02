import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, average_precision_score)
import os
import gc
import joblib

DATA_PATH = "./data_pipeline/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TabTransformer 모델 구조 정의 (파일 내부에 포함)
class TabTransformer(nn.Module):
    def __init__(self, n_cont, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        # 수치형 변수 임베딩
        self.cont_embed = nn.Linear(n_cont, n_cont * embed_dim)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 최종 분류 헤드
        self.mlp = nn.Sequential(
            nn.Linear(n_cont * embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        # (batch, n_cont) -> (batch, n_cont, embed_dim)
        x = self.cont_embed(x).view(batch_size, -1, 32)
        x = self.transformer(x)
        x = x.view(batch_size, -1) # Flatten
        return self.mlp(x)

def evaluate_tt():
    data_files = [("SMOTE", "train_smote.csv"), ("cGAN", "train_cgan.csv"), ("K-cGAN", "train_kcgan.csv")]
    final_results = []

    for name, file_name in data_files:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path): continue

        print(f"🚀 [{name}] TabTransformer 학습 및 평가 중...")
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        # 8:2 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # 전처리 (Standard Scaling)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 데이터 로더 준비
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s).to(device), 
                                          torch.FloatTensor(y_train).view(-1, 1).to(device)), 
                            batch_size=1024, shuffle=True)

        # 모델 생성
        model = TabTransformer(n_cont=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # 학습 (30 Epoch)
        model.train()
        for epoch in range(30):
            for bx, by in loader:
                optimizer.zero_grad()
                output = model(bx)
                loss = criterion(output, by)
                loss.backward(); optimizer.step()

        # 평가
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_s).to(device)
            probs = model(X_test_tensor).cpu().numpy()
            preds = (probs > 0.5).astype(int)

        final_results.append({
            "Method": name,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs)
        })
        del df, X_train, X_test; gc.collect()

    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 TabTransformer 기반 증강 기법별 성능 비교 (8:2 Split)")
    print("="*70)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_tt()