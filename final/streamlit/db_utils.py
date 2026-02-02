# db_utils.py
import pymysql
from datetime import date

def save_metrics_to_mysql(
    report_df,
    model_name: str,
    host="127.0.0.1",
    port=3306,
    user="root",
    password="zxcv1234",
    db="threat_intel"
):
    """
    report_df는 아래 컬럼을 가지고 있어야 함:
    Method, Precision, Recall, F1-Score, ROC-AUC, AUPRC
    """
    conn = pymysql.connect(
        host=host, port=port, user=user, password=password,
        db=db, charset="utf8mb4", autocommit=True
    )
    try:
        cur = conn.cursor()

        # 테이블이 없으면 만들어주기(안전장치)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fds_model_metrics_daily (
          metric_date DATE NOT NULL,
          model_name VARCHAR(50) NOT NULL,
          method VARCHAR(30) NOT NULL,
          precision_val DOUBLE,
          recall_val DOUBLE,
          f1_val DOUBLE,
          roc_auc_val DOUBLE,
          auprc_val DOUBLE,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (metric_date, model_name, method)
        )
        """)

        for _, r in report_df.iterrows():
            cur.execute("""
                INSERT INTO fds_model_metrics_daily
                (metric_date, model_name, method, precision_val, recall_val, f1_val, roc_auc_val, auprc_val)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  precision_val=VALUES(precision_val),
                  recall_val=VALUES(recall_val),
                  f1_val=VALUES(f1_val),
                  roc_auc_val=VALUES(roc_auc_val),
                  auprc_val=VALUES(auprc_val)
            """, (
                date.today(),
                model_name,
                str(r["Method"]),
                float(r["Precision"]),
                float(r["Recall"]),
                float(r["F1-Score"]),
                float(r["ROC-AUC"]),
                float(r["AUPRC"]),
            ))
    finally:
        conn.close()
