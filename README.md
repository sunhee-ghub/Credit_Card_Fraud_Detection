# 💳 Credit Card Fraud Detection using K-CGAN & Deep Learning

> **데이터 불균형 해소를 위한 K-Means 기반 Conditional GAN 증강 및 딥러닝 모델 탐지 성능 비교 연구**

본 프로젝트는 신용카드 거래 데이터의 극심한 불균형 문제를 해결하기 위해, 논문에서 제안된 **K-CGAN(K-Means Conditional GAN)** 기법을 1:1 균형 데이터셋으로 증강하여 최신 딥러닝 모델들의 탐지 성능(F1-Score, Recall)을 극대화하는 것을 목표로 합니다.

---

## 📑 1. 프로젝트 배경 (Background)
* **문제점**: 신용카드 사기 데이터는 정상 거래 대비 사기 거래 비율이 0.17%에 불과하여, 일반적인 모델 학습 시 다수 클래스(정상)에 편향된 결과가 도출됨.
* **기존 기법의 한계**: SMOTE와 같은 단순 보간법은 데이터의 비선형적인 패턴이나 고차원적인 특징을 생성하는 데 한계가 있음.
* **해결 방안**: GAN을 활용하여 실제 사기 데이터의 분포를 학습하고, 특히 사기 데이터 내의 다양한 패턴을 반영하기 위해 **K-Means 클러스터링을 결합한 K-CGAN** 기법을 적용함.



---

## 🛠 2. 데이터 증강 방법론 (Data Augmentation)

본 프로젝트는 총 3가지 형태의 학습 데이터를 준비하여 비교 분석합니다.

### 2.1 SMOTE (Synthetic Minority Over-sampling Technique)
* 소수 클래스 샘플 간의 거리를 기반으로 가상의 샘플을 생성하는 전통적인 오버샘플링 기법.

### 2.2 cGAN (Conditional GAN)
* 사기 데이터 여부(Class)를 조건(Condition)으로 주어 데이터의 특징을 학습하고 생성하는 모델.

### 2.3 K-CGAN (Novel Approach)
* **단계 1 (Clustering)**: 사기 데이터들을 K-Means 알고리즘을 통해 $K$개의 군집으로 나눔. (사기 유형별 특징 추출)
* **단계 2 (Conditioning)**: 각 군집 레이블을 cGAN의 조건으로 입력하여, 특정 사기 유형의 분포를 정교하게 학습함.
* **단계 3 (Generation)**: 학습된 생성자가 각 군집별로 고품질의 가짜 사기 샘플을 생성함.

---

## 🧪 3. 실험 설계 (Experiment Design)

분석가로서 데이터 증강 효율을 검증하기 위해 **4 x 4 실험 매트릭스**를 구성하여 총 12번의 독립적인 실험을 수행합니다.

### 3.1 실험 매트릭스
| 분류 모델 \ 데이터셋 | SMOTE | cGAN | **K-CGAN** |
| :--- | :---: | :---: | :---: |
| **Random Forest** | Exp 1 | Exp 5 | Exp 9 |
| **TabNet** | Exp 2 | Exp 6 | Exp 10 |
| **BERT (Tabular)** | Exp 3 | Exp 7 | Exp 11 |
| **AutoEncoder** | Exp 4 | Exp 8 | Exp 12 |

### 3.2 평가 지표 (Metrics)
단순 정확도(Accuracy)는 불균형 데이터에서 무의미하므로 아래 지표를 중점적으로 평가합니다.
* **Recall (재현율)** : 실제 사기를 얼마나 놓치지 않고 잡아냈는가.
* **Precision (정밀도)** : 사기라고 예측한 것 중 실제 사기가 얼마나 있는가.
* **F1-Score** : Precision과 Recall의 조화 평균. (가장 핵심 지표)
* **ROC-AUC** : 모델의 전반적인 분류 변별력.
* **AUPRC** : 지표를 통해 사기 탐지의 정밀도.



---

## 💻 4. 모델 아키텍처 (Model Architecture)

### 4.1 TabNet
* 정형 데이터에 최적화된 Attention 기반 모델로, 피처 선택 기능을 내장하여 중요한 변수(V1~V28)에 집중함.

### 4.2 BERT for Tabular (TabTransformer)
* 각 변수를 임베딩하여 수치형 데이터 간의 상관관계를 Transformer 레이어를 통해 학습함.

### 4.3 AutoEncoder
* 정상 데이터를 복원하도록 학습하여, 복원 오차가 큰 샘플을 사기로 간주. 증강 데이터를 활용하여 Threshold를 최적화함.

---

## 📈 5. 주요 성과 및 시각화 (Results)

* **t-SNE 시각화 분석**: K-CGAN을 통해 생성된 사기 데이터가 정상 거래 분포와 뚜렷한 거리를 두며, **독자적인 클러스터링을 형성**하고 있음을 기술적으로 검증함.
![t-SNE Visualization Result](https://github.com/user-attachments/assets/d79dd503-922d-4646-89ff-dad8f5208015)
*그림 1. 원본 데이터와 K-CGAN 생성 데이터의 t-SNE 시각화 비교 (정상 거래와의 변별력 확인)*

* **데이터 정합성**: 생성된 데이터가 단순 복제가 아닌, 원본 사기 패턴의 통계적 특성을 유지하면서 새로운 변동성을 가짐을 확인.
* **모델 신뢰도**: 1:1 균형 데이터셋 학습을 통해 모든 모델에서 Recall과 F1-Score의 유의미한 향상을 기록.

---
* **모델 신뢰도 확보**: 1:1 균형 데이터셋 학습을 통해 모든 모델에서 Recall과 F1-Score의 유의미한 향상을 기록했으며, 특히 **AUPRC 지표를 통해 사기 탐지의 정밀도**를 입증함.
* **논리적 타당성**: 8:2 독립 검증을 통해 가상 데이터가 단순 복제가 아닌, 모델이 학습 가능한 유의미한 패턴을 가지고 있음을 증명함.

---

## 📂 6. 프로젝트 구조 (Directory Structure)
```text
├── final/
│   ├── Preprocessing.py    # 데이터 분포 확인 및 전처리
│   ├── Augmentation_SMOTE.py       # 데이터 증강(SMOTE)
│   ├── Augmentation_cCAN.py        # 데이터 증강(cGAN)
│   └── Augmentation_K_cGAN.py      # 데이터 증강(K-cGAN)
├── models/
│   ├── randomforest.py             # 랜덤포레스트 모델
│   ├── tabtransformer.py           # BERT(tabtransformer) 모델
│   ├── autoencoder.py              # autoencoder 모델
│   └── tabnet.py                   # tabnet 모델
├── visualization/
│   └── visualization.ipynb         # t_SNE 시각화
├── results/
│   ├── randomforest_result.csv     # 랜덤포레스트 모델 지표
│   ├── tabtransformer_result.csv   # BERT(tabtransformer) 모델 지표
│   ├── autoencoder_result.csv      # autoencoder 모델 지표
│   └── tabnet_result.csv           # tabnet 모델 지표
└── README.md
