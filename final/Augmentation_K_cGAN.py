import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 전역 설정 (하이퍼파라미터) ---
N_CLUSTERS = 5     # 사기 패턴 군집 수
LATENT_DIM = 100   # 노이즈 차원
EPOCHS = 300       # 학습 횟수 (로컬 사양에 따라 100~300 조절)
BATCH_SIZE = 1024  # 로컬 메모리를 고려해 1024로 조정 가능
KL_WEIGHT = 0.1    # KL Loss 가중치

# 1. KL Divergence Loss 정의
def compute_kl_loss(real_samples, fake_samples):
    mu_real = real_samples.mean(dim=0)
    sigma_real = real_samples.var(dim=0) + 1e-6
    mu_fake = fake_samples.mean(dim=0)
    sigma_fake = fake_samples.var(dim=0) + 1e-6
    
    kl = 0.5 * (torch.log(sigma_fake / sigma_real) + 
                (sigma_real + (mu_real - mu_fake)**2) / sigma_fake - 1)
    return kl.sum()

# 2. 모델 아키텍처
class K_Generator(nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim):
        super(K_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh() 
        )
    def forward(self, noise, cond):
        x = torch.cat([noise, cond], 1)
        return self.model(x)

class K_Discriminator(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super(K_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + cond_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, data, cond):
        x = torch.cat([data, cond], 1)
        return self.model(x)

# 3. 메인 실행 함수
def run_final_kcgan_local():
    print("🚀 [K-cGAN + KL Loss] 로컬 학습 시작...")
    
    # 데이터 로드 (파일 경로가 현재 폴더에 있어야 함)
    file_name = "base_preprocessed.csv"
    if not os.path.exists(file_name):
        print(f"❌ Error: '{file_name}' 파일이 없습니다. 경로를 확인하세요.")
        return
    df = pd.read_csv(file_name)
    
    # 데이터 분할
    train_df, test_df_real = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']

    # 스케일링 및 클리핑
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.clip(X_train_scaled / 5.0, -1, 1)

    # K-means 군집화
    fraud_mask = (y_train == 1).values
    X_fraud = X_train_scaled[fraud_mask]
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    fraud_clusters = kmeans.fit_predict(X_fraud)
    
    # 조건 벡터 생성
    cluster_labels = np.zeros(len(X_train_scaled))
    cluster_labels[fraud_mask] = fraud_clusters + 1
    cond_matrix = pd.get_dummies(cluster_labels).values
    cond_dim = cond_matrix.shape[1]

    # 장치 설정 (NVIDIA GPU 우선)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 사용 장치: {device}")

    # 모델 생성
    G = K_Generator(LATENT_DIM, cond_dim, X_train_scaled.shape[1]).to(device)
    D = K_Discriminator(X_train_scaled.shape[1], cond_dim).to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_scaled).to(device), 
        torch.FloatTensor(cond_matrix).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 학습 루프
    for epoch in range(EPOCHS):
        G.train(); D.train()
        for real_data, conds in loader:
            b_size = real_data.size(0)
            
            # D 학습
            optimizer_D.zero_grad()
            real_out = D(real_data, conds)
            d_loss_real = criterion(real_out, torch.ones(b_size, 1).to(device) * 0.9)
            
            z = torch.randn(b_size, LATENT_DIM).to(device)
            fake_data = G(z, conds)
            fake_out = D(fake_data.detach(), conds)
            d_loss_fake = criterion(fake_out, torch.zeros(b_size, 1).to(device))
            
            (d_loss_real + d_loss_fake).backward()
            optimizer_D.step()

            # G 학습 (BCE + KL)
            optimizer_G.zero_grad()
            g_out = D(fake_data, conds)
            g_loss_bce = criterion(g_out, torch.ones(b_size, 1).to(device))
            g_loss_kl = compute_kl_loss(real_data, fake_data)
            
            total_g_loss = g_loss_bce + (KL_WEIGHT * g_loss_kl)
            total_g_loss.backward()
            optimizer_G.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] D_Loss: {d_loss_real.item()+d_loss_fake.item():.4f} | G_KL: {g_loss_kl.item():.4f}")

    # 데이터 생성 및 저장
    print("🪄 합성 데이터 생성 및 파일 저장 중...")
    num_to_gen = sum(y_train == 0) - sum(y_train == 1)
    unique, counts = np.unique(fraud_clusters, return_counts=True)
    cluster_probs = counts / counts.sum()
    gen_cluster_ids = np.random.choice(np.arange(1, N_CLUSTERS+1), size=num_to_gen, p=cluster_probs)
    
    gen_cond_matrix = np.zeros((num_to_gen, cond_dim))
    for i, cid in enumerate(gen_cluster_ids): gen_cond_matrix[i, cid] = 1.0

    z = torch.randn(num_to_gen, LATENT_DIM).to(device)
    G.eval()
    with torch.no_grad():
        gen_samples = G(z, torch.FloatTensor(gen_cond_matrix).to(device)).cpu().numpy()

    gen_samples = scaler.inverse_transform(gen_samples * 5.0)
    df_gen = pd.DataFrame(gen_samples, columns=X_train.columns)
    df_gen['Class'] = 1
    
    # 최종 파일 저장
    df_train_final = pd.concat([train_df[train_df['Class']==0], df_gen], axis=0)
    df_train_final.to_csv("train_kcgan.csv", index=False)
    
    print(f"✅ 파일 생성 완료: {os.getcwd()}\\train_kcgan.csv")

if __name__ == "__main__":
    run_final_kcgan_local()