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
        if not os.path.exists(path): continue

        print(f"🚀 [{name}] RF 학습 및 평가 중...")
        df = pd.read_csv(path)
        X = df.drop('Class', axis=1).values.astype(np.float32)
        y = df['Class'].values.astype(int)

        # 8:2 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # 모델 학습 (Random Forest)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 평가
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        final_results.append({
            "Method": name,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs)
        })
        del df, X_train, X_test; gc.collect()

    report_df = pd.DataFrame(final_results)
    print("\n" + "="*70)
    print("📊 Random Forest 기반 증강 기법별 성능 비교 (8:2 Split)")
    print("="*70)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_rf()