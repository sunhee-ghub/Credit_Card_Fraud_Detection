import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "/opt/airflow/data_pipeline/train_smote.csv"
OUT_PATH = "/opt/airflow/models/randomforest.joblib"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1).values
y = df["Class"].astype(int).values

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)
joblib.dump(model, OUT_PATH)

print("✅ RandomForest saved:", OUT_PATH)
