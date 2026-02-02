import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import os

# ===============================
# 1. 경로 및 장치 설정
# ===============================
DATA_PATH = "./data_pipeline/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. Deep AutoEncoder 정의
# ===============================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3), nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 24), nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def evaluate_ae():
    methods = [
        ("SMOTE", "train_smote.csv"),
        ("cGAN", "train_cgan.csv"),
        ("K-cGAN", "train_kcgan.csv")
    ]

    final_results = []

    for name, file_name in methods:
        print(f"\n🚀 {name} (Deep AE) 학습 및 평가 시작...")

        file_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ {file_name} 파일이 없어 건너뜁니다.")
            continue

        # ===============================
        # 데이터 로드
        # ===============================
        df = pd.read_csv(file_path)
        X = df.drop("Class", axis=1).values
        y = df["Class"].values

        # ✅ AE 필수: 스케일링
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # ===============================
        # Train / Test 분리
        # ===============================
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 정상 데이터만 학습
        X_train_normal = X_train_all[y_train_all == 0]

        # ===============================
        # 모델 설정
        # ===============================
        model = AutoEncoder(X_train_normal.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_normal)),
            batch_size=256,
            shuffle=True
        )

        # ===============================
        # 모델 학습
        # ===============================
        model.train()
        for epoch in range(100):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

        # ===============================
        # 평가 (Reconstruction Error)
        # ===============================
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            reconstructed = model(X_test_tensor)
            mse_scores = torch.mean(
                (X_test_tensor - reconstructed) ** 2, dim=1
            ).cpu().numpy()

        # ===============================
        # Threshold 탐색
        # ===============================
        precisions, recalls, thresholds = precision_recall_curve(
            y_test, mse_scores
        )

        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]

        preds = (mse_scores > best_threshold).astype(int)

        final_results.append({
            "Method": name,
            "Threshold": round(float(best_threshold), 6),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "ROC-AUC": round(roc_auc_score(y_test, mse_scores), 4),
            "AUPRC": round(average_precision_score(y_test, mse_scores), 4)
        })

        print(f"✅ {name} 완료 (F1={final_results[-1]['F1-Score']})")

    # ===============================
    # 결과 정리 + DB 저장
    # ===============================
    results_df = pd.DataFrame(final_results)
    print("\n📊 Deep AutoEncoder 최종 결과")
    print(results_df)

    from db_utils import save_metrics_to_mysql
    save_metrics_to_mysql(results_df, model_name="AutoEncoder")

    print("💾 DB 저장 완료 (기존 AutoEncoder 결과는 최신 값으로 갱신됨)")


if __name__ == "__main__":
    evaluate_ae()
