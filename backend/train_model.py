import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("../dataset/heart_rate_data.csv")

print(df.columns)

# Feature Engineering
df["recovery"] = df["Resting Heart Rate Before"] - \
    df["Resting Heart Rate After"]
df["mhr"] = 220 - df["Age"]
df["stress"] = df["Max Heart Rate During Exercise"] / df["mhr"]

df["S_RHR"] = 1 - ((df["Resting Heart Rate Before"] - 50) / 50)
df["S_REC"] = df["recovery"] / df["Resting Heart Rate Before"]
df["S_STR"] = 1 - df["stress"]
df["S_SLP"] = df["Sleep Hours"] / 8

for col in ["S_RHR", "S_REC", "S_STR", "S_SLP"]:
    df[col] = df[col].clip(0, 1)

df["health_score"] = 100 * (
    0.25 * df["S_RHR"] +
    0.25 * df["S_REC"] +
    0.25 * df["S_STR"] +
    0.25 * df["S_SLP"]
)

df["health_score"] = df["health_score"] * 1.5
df["health_score"] = df["health_score"].clip(0, 100)

# Label


def create_label(row):
    if row["recovery"] < 5 or row["stress"] > 0.85 or row["Sleep Hours"] < 6:
        return 1
    else:
        return 0


df["label"] = df.apply(create_label, axis=1)

X = df[["Age", "Sleep Hours", "recovery", "stress"]]
y_risk = df["label"]
y_score = df["health_score"]

# Models
risk_model = LogisticRegression(max_iter=1000)
risk_model.fit(X, y_risk)

score_model = RandomForestRegressor(n_estimators=100, random_state=42)
score_model.fit(X, y_score)

# Save models
joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(score_model, "score_model.pkl")

print("✅ Models trained successfully")
