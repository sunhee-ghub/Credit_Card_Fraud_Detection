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

# 1. 경로 설정
DATA_PATH = "./data_pipeline/"
RESULT_PATH = "./results/"
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Deep AutoEncoder 모델 정의 (레이어 2개씩 추가)
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder: 입력 -> 24 -> 16 -> 8 -> 3 (점진적 압축)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),  # 추가된 레이어 1
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),  # 추가된 레이어 2
            nn.Linear(8, 3), nn.ReLU()  # 최종 Bottleneck (3차원)
        )

        # Decoder: 3 -> 8 -> 16 -> 24 -> 입력 (대칭 복원)
        self.decoder = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(),  # 추가된 레이어 1
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 24), nn.ReLU(),  # 추가된 레이어 2
            nn.Linear(24, input_dim)  # 최종 출력 (활성화 함수 없음)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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

        # 데이터 로드
        df = pd.read_csv(file_path)
        X = df.drop('Class', axis=1).values
        y = df['Class'].values

        # 8:2 분할
        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 정상 데이터만 학습에 사용
        X_train_normal = X_train_all[y_train_all == 0]

        # 모델 초기화
        model = AutoEncoder(X_train_normal.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 학습률 유지
        criterion = nn.MSELoss()

        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_normal).to(device)),
                            batch_size=256, shuffle=True)

        # 학습 (모델이 깊어졌으므로 Epoch 유지하거나 필요시 상향)
        model.train()
        for epoch in range(100):
            for [batch] in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

        # 평가
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            reconstructed = model(X_test_tensor)
            mse_scores = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        # 최적 임계치 탐색
        precisions, recalls, thresholds = precision_recall_curve(y_test, mse_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]

        preds = (mse_scores > best_threshold).astype(int)

        # 결과 저장
        res = {
            "Method": name,
            "Threshold": round(float(best_threshold), 6),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "ROC-AUC": round(roc_auc_score(y_test, mse_scores), 4),
            "AUPRC": round(average_precision_score(y_test, mse_scores), 4)
        }
        final_results.append(res)
        print(f"✅ {name} 완료: F1={res['F1-Score']}, AUPRC={res['AUPRC']}")

    # CSV 저장
    if final_results:
        results_df = pd.DataFrame(final_results)
        save_path = os.path.join(RESULT_PATH, "autoencoder_deep_results.csv")
        results_df.to_csv(save_path, index=False)

        print("\n" + "=" * 80)
        print("📊 Deep AutoEncoder 최종 성능 리포트")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        print(f"💾 결과 저장 완료: {save_path}")


if __name__ == "__main__":
    evaluate_ae()