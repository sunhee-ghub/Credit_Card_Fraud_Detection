import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import gc
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, recall_score, precision_score

# --- [1. 장치 설정 및 공통 데이터 로드] ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_variants = ['org', 'smote', 'cgan', 'kcgan']

print(f"🔍 테스트 데이터를 로드 중... (Device: {device})")
X_test_scaled = joblib.load('X_test_scaled.pkl')
y_test = joblib.load('y_test.pkl')
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

# --- [2. TabTransformer 모델 아키텍처 정의] ---
# Transformer Encoder를 사용하여 피처 간의 상호작용을 학습합니다.
class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TabTransformer, self).__init__()
        self.embed_dim = embed_dim
        # 수치형 피처를 임베딩 공간으로 투사
        self.input_projection = nn.Linear(input_dim, input_dim * embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 이진 분류를 위한 출력층
        )

    def forward(self, x):
        # [Batch, Input_Dim] -> [Batch, Input_Dim, Embed_Dim]
        x = self.input_projection(x).view(x.size(0), -1, self.embed_dim)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)

# --- [3. 모델 학습 및 평가 루프] ---
tab_results = []

for variant in data_variants:
    print(f"\n" + "="*50)
    print(f"🚀 TabTransformer 학습 시작: [{variant.upper()}] 데이터셋")
    print("="*50)
    
    # 1) 데이터셋 로드
    X_tr = joblib.load(f'X_train_{variant}.pkl')
    y_tr = joblib.load(f'y_train_{variant}.pkl')
    
    input_dim = X_tr.shape[1]
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).view(-1, 1)), 
        batch_size=1024, shuffle=True
    )
    
    # 2) 모델 및 최적화 설정
    # pos_weight=4.0: 사기 데이터(1)에 4배의 가중치를 두어 불균형 해소
    model = TabTransformer(input_dim, embed_dim=32, num_heads=8, num_layers=3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 3) 학습 (Training)
    model.train()
    for epoch in range(30): # 30 에포크 학습
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/30] - Loss: {total_loss/len(train_loader):.4f}")

    # 4) 평가 (Evaluation)
    model.eval()
    with torch.no_grad():
        # 예측 확률 계산 (Sigmoid 적용)
        logits = model(X_test_tensor)
        tab_probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Precision-Recall Curve를 통해 최적의 F1-Score와 임계값 찾기
    precisions, recalls, thresholds = precision_recall_curve(y_test, tab_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    tab_results.append({
        "Dataset": variant.upper(),
        "F1-Score": f1_scores[best_idx],
        "Recall": recalls[best_idx],
        "Precision": precisions[best_idx],
        "ROC-AUC": roc_auc_score(y_test, tab_probs),
        "Best_Threshold": thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    })

    # 5) 메모리 정리
    del X_tr, y_tr, model, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- [4. 최종 결과 출력] ---
print("\n" + "✨" * 25)
print("🏆 TabTransformer 최종 실험 결과 리포트")
print("✨" * 25)
tab_df = pd.DataFrame(tab_results)
pd.options.display.float_format = '{:.4f}'.format
print(tab_df.sort_values(by="F1-Score", ascending=False).to_string(index=False))