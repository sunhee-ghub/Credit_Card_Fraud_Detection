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

# ✅ 속도/시간 조절 파라미터 (발표용 추천 값)
EPOCHS = 5                 # 30 -> 5
BATCH_SIZE = 512           # 1024 -> 512
MAX_TRAIN_ROWS = 100_000   # SMOTE 45만 행 -> 10만 행만 학습 (속도 확 줄어듦)
EMBED_DIM = 16             # 32 -> 16 (가벼워짐)
N_LAYERS = 1               # 2 -> 1
N_HEADS = 2                # 4 -> 2

class TabTransformer(nn.Module):
    def __init__(self, n_cont, embed_dim=16, n_heads=2, n_layers=1):
        super().__init__()
        self.n_cont = n_cont
        self.embed_dim = embed_dim

        self.cont_embed = nn.Linear(n_cont, n_cont * embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(n_cont * embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size(0)
        x = self.cont_embed(x).view(b, self.n_cont, self.embed_dim)
        x = self.transformer(x)
        x = x.reshape(b, -1)
        return self.mlp(x)

def evaluate_tt():
    data_files = [("SMOTE", "train_smote.csv"), ("cGAN", "train_cgan.csv"), ("K-cGAN", "train_kcgan.csv")]
    final_results = []

    for name, file_name in data_files:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            print(f"❌ 파일 없음: {path}")
            continue

        print(f"🚀 [{name}] TabTransformer 학습 및 평가 중...")

        df = pd.read_csv(path)

        # ✅ (중요) 너무 크면 샘플링해서 시간 줄이기
        if len(df) > MAX_TRAIN_ROWS:
            df = df.sample(n=MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)
            print(f"   ✅ 샘플링 적용: {MAX_TRAIN_ROWS} rows만 사용")

        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        )
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = TabTransformer(
            n_cont=X_train.shape[1],
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
            n_layers=N_LAYERS
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(EPOCHS):
            running = 0.0
            for step, (bx, by) in enumerate(loader, start=1):
                bx = bx.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running += loss.item()

                # ✅ 진행 표시 (안 멈춘 거 확인 가능)
                if step % 50 == 0:
                    print(f"   - epoch {epoch+1}/{EPOCHS} step {step}/{len(loader)} loss {loss.item():.6f}")

            print(f"✅ epoch {epoch+1}/{EPOCHS} avg_loss={running/len(loader):.6f}")

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32).to(device)
            probs = model(X_test_tensor).detach().cpu().numpy().reshape(-1)  # (N,)
            preds = (probs > 0.5).astype(int)

        final_results.append({
            "Method": name,
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs)
        })

        # ✅ 저장 (torch 방식)
        torch.save(model.state_dict(), f"tt_{name.lower()}_state.pt")
        joblib.dump(scaler, f"tt_{name.lower()}_scaler.pkl")

        del df, X, y, X_train, X_test, X_train_s, X_test_s, train_ds, loader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 TabTransformer 기반 증강 기법별 성능 비교")
    print("="*70)
    print(report_df.to_string(index=False))

    from db_utils import save_metrics_to_mysql
    save_metrics_to_mysql(report_df, model_name="TabTransformer")

if __name__ == "__main__":
    evaluate_tt()
