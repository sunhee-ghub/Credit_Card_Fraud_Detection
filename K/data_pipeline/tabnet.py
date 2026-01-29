import pandas as pd
import numpy as np
import torch
import gc
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, 
                             recall_score, average_precision_score)

# [설정] 데이터 경로 및 파일 구성
DATA_PATH = "./data_pipeline/"
data_files = [
    ("SMOTE", "train_smote.csv"),
    ("cGAN", "train_cgan.csv"),
    ("K-cGAN", "train_kcgan.csv")
]

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_tabnet():
    final_results = []

    for name, file_name in data_files:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            continue

        print(f"🚀 [{name}] 데이터 처리 중...")
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        # 8:2 분할 (층화 추출 적용)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 모델 정의 및 학습
        clf = TabNetClassifier(device_name=device_name, verbose=0)
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['auc'],
            max_epochs=20, patience=5,
            batch_size=1024, virtual_batch_size=128
        )

        # 예측 (Probability는 AUC/AUPRC 계산용)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        # 지표 계산
        res = {
            "Method": name,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs) # 핵심 지표 추가
        }
        final_results.append(res)
        
        del X, y, X_train, X_test, df; gc.collect()

    # 결과 출력
    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("🏆 TabNet 기반 증강 기법별 성능 비교 (8:2 Split)")
    print("="*70)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_tabnet()