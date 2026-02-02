import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

# --- 로컬 경로 설정 ---
# 원본 파일이 있는 폴더 경로를 지정하세요. 
# 파일이 코드와 같은 폴더에 있다면 ""으로 두시면 됩니다.
INPUT_FILE = "base_preprocessed.csv" 
OUTPUT_FILE = "train_cgan.csv"

# 1. 모델 정의
class Generator(nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], 1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, data, labels):
        x = torch.cat([data, labels], 1)
        return self.model(x)

# 2. 메인 실행 함수
def run_cgan_local_augmentation():
    print(f"🚀 로컬 작업 시작: {INPUT_FILE} 로딩 중...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 에러: {INPUT_FILE} 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(INPUT_FILE)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # NVIDIA GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 현재 사용 장치: {device}")
    
    latent_dim, data_dim, cond_dim = 100, X.shape[1], 1
    G = Generator(latent_dim, cond_dim, data_dim).to(device)
    D = Discriminator(data_dim, cond_dim).to(device)
    optim_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X.values).to(device)
    y_tensor = torch.FloatTensor(y.values).view(-1, 1).to(device)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tensor, y_tensor), batch_size=1024, shuffle=True)

    print("🛠️ cGAN 학습 진행 중 (100 Epochs)...")
    for epoch in range(100):
        G.train(); D.train()
        for real_data, labels in loader:
            b_size = real_data.size(0)
            
            # Discriminator 학습
            optim_D.zero_grad()
            real_loss = criterion(D(real_data, labels), torch.ones(b_size, 1).to(device) * 0.9)
            z = torch.randn(b_size, latent_dim).to(device)
            fake_data = G(z, labels)
            fake_loss = criterion(D(fake_data.detach(), labels), torch.zeros(b_size, 1).to(device))
            (real_loss + fake_loss).backward()
            optim_D.step()

            # Generator 학습
            optim_G.zero_grad()
            g_loss = criterion(D(fake_data, labels), torch.ones(b_size, 1).to(device))
            g_loss.backward()
            optim_G.step()
        
        if (epoch+1) % 20 == 0:
            print(f"   [{epoch+1}/100] D Loss: {real_loss.item()+fake_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    print("🪄 1:1 비율을 맞추기 위한 데이터 생성 중...")
    num_gen = sum(y == 0) - sum(y == 1) # 정상 데이터 수만큼 사기 데이터 생성
    
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_gen, latent_dim).to(device)
        cond = torch.ones(num_gen, 1).to(device) 
        gen_samples = G(z, cond).cpu().numpy()
    
    df_gen = pd.DataFrame(gen_samples, columns=X.columns)
    df_gen['Class'] = 1
    
    # 데이터 병합
    df_final = pd.concat([df, df_gen], axis=0)
    print(f"📊 최종 클래스 비율:\n{df_final['Class'].value_counts()}")

    # 로컬 저장
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 저장 완료: {os.getcwd()}\\{OUTPUT_FILE}")

if __name__ == "__main__":
    run_cgan_local_augmentation()