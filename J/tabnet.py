import pandas as pd
import numpy as np
import torch
import gc
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, average_precision_score)

# 1. 경로 설정 및 결과 저장 폴더 생성
DATA_PATH = "./data_pipeline/"
RESULT_PATH = "./results/"
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# 데이터 파일 리스트
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

        print(f"\n🚀 [{name}] TabNet 학습 및 평가 시작...")

        # 데이터 로드
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        # 2. 8:2 분할 (층화 추출 적용)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 3. TabNet 모델 정의 및 학습
        # N_d, N_a는 모델의 복잡도를 결정하며, 정형 데이터에서는 8~64 사이가 적당합니다.
        clf = TabNetClassifier(
            n_d=16, n_a=16,  # Attention 레이어 크기
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax',  # 가독성 높은 피처 선택을 위해 sparsemax 사용
            device_name=device_name,
            verbose=0
        )

        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['auc'],
            max_epochs=50,  # 학습 횟수 상향
            patience=10,  # 성능 개선 없을 시 조기 종료
            batch_size=1024, virtual_batch_size=128
        )

        # 4. 예측 및 지표 계산
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        # 결과 딕셔너리 생성
        res = {
            "Method": name,
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
            "AUPRC": round(average_precision_score(y_test, probs), 4)
        }
        final_results.append(res)
        print(f"✅ {name} 완료: AUPRC={res['AUPRC']}, F1={res['F1-Score']}")

        # 메모리 관리
        del clf, X_train, X_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. 최종 결과를 CSV로 저장 및 출력
    if final_results:
        results_df = pd.DataFrame(final_results)

        # 파일 저장 경로
        save_path = os.path.join(RESULT_PATH, "tabnet_performance_results.csv")
        results_df.to_csv(save_path, index=False)

        print("\n" + "=" * 80)
        print("📊 TabNet 최종 성능 비교 리포트")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        print(f"💾 결과가 '{save_path}'에 저장되었습니다.")
    else:
        print("❌ 분석된 결과가 없습니다.")


if __name__ == "__main__":
    evaluate_tabnet()