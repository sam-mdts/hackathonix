import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("all_data.csv")
X = df.drop(columns=["person", "label"]).values
y = df["label"].values

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
model.fit(X, y)
joblib.dump(model, "model.pkl")
print(f"Trained on {len(df)} rows from {df['person'].nunique()} people. Saved model.pkl")