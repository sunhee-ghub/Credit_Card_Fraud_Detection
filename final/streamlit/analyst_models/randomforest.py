import pandas as pd
import numpy as np
import os
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, average_precision_score)

DATA_PATH = "./data_pipeline/"

def evaluate_rf():
    data_files = [("SMOTE", "train_smote.csv"), ("cGAN", "train_cgan.csv"), ("K-cGAN", "train_kcgan.csv")]
    final_results = []

    for name, file_name in data_files:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            continue

        print(f"🚀 [{name}] RF 학습 및 평가 중...")

        # ✅ 1) 읽을 때부터 dtype 줄이기
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).astype(np.float32).to_numpy()
        y = df['Class'].astype(int).to_numpy()

        # 8:2 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ✅ 2) 메모리 줄이는 모델 옵션
        model = RandomForestClassifier(
            n_estimators=80,       # 100 -> 80 (조금 줄임)
            max_depth=12,         # ✅ 트리 깊이 제한(메모리/시간 크게 줄어듦)
            min_samples_leaf=2,   # ✅ 잎 최소 샘플
            n_jobs=1,             # ✅ 병렬 금지(가장 중요)
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        final_results.append({
            "Method": name,
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs)
        })

        joblib.dump(model, f"rf_{name.lower()}.pkl")

        # ✅ 3) 진짜로 메모리 확실히 비우기
        del df, X, y, X_train, X_test, y_train, y_test, model, preds, probs
        gc.collect()

    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 Random Forest 기반 증강 기법별 성능 비교 (8:2 Split)")
    print("="*70)
    print(report_df.to_string(index=False))

    from db_utils import save_metrics_to_mysql
    save_metrics_to_mysql(report_df, model_name="RandomForest")

if __name__ == "__main__":
    evaluate_rf()
