"""Train the churn model and precompute dashboard statistics."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv")
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT = ["Geography", "Gender"]
CATS = {c: sorted(df[c].unique()) for c in CAT}


def encode(X):
    X = X[FEATURES].copy()
    for c in CAT:
        X[c] = X[c].astype("category").cat.set_categories(CATS[c])
    return X


tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Exited"])
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                       categorical_features="from_dtype", random_state=42)
model.fit(encode(tr), tr["Exited"])

p = model.predict_proba(encode(te))[:, 1]
y = te["Exited"].values
auc = roc_auc_score(y, p)
fpr, tpr, _ = roc_curve(y, p)
idx = np.linspace(0, len(fpr) - 1, 60).astype(int)
tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel()
hist_s = np.histogram(p[y == 0], bins=20, range=(0, 1))[0] / (y == 0).sum()
hist_l = np.histogram(p[y == 1], bins=20, range=(0, 1))[0] / (y == 1).sum()


def rate(col, bins=None, labels=None):
    s = pd.cut(df[col], bins, labels=labels, right=False) if bins is not None else df[col]
    g = df.groupby(s, observed=True)["Exited"].agg(["mean", "size"])
    return [{"label": str(k), "rate": round(float(r["mean"]) * 100, 1), "n": int(r["size"])}
            for k, r in g.iterrows()]


act = rate("IsActiveMember")
for r in act:
    r["label"] = "Active" if r["label"] in ("1", "1.0") else "Inactive"

insights = {
    "Age": rate("Age", [18, 30, 40, 50, 60, 120], ["18-29", "30-39", "40-49", "50-59", "60+"]),
    "Products": rate("NumOfProducts"),
    "Country": rate("Geography"),
    "Activity": act,
    "Credit score": rate("CreditScore", [0, 500, 600, 700, 800, 900], ["<500", "500-599", "600-699", "700-799", "800+"]),
    "Tenure": rate("Tenure", [0, 1, 3, 5, 8, 11], ["<1y", "1-2y", "3-4y", "5-7y", "8-10y"]),
    "Gender": rate("Gender"),
}

bg = df.sample(300, random_state=1)[FEATURES]
cust = te.sample(400, random_state=7).copy()
cust["prob"] = model.predict_proba(encode(cust))[:, 1]


def py(v):
    return v.item() if hasattr(v, "item") else v


cust_rows = [{**{c: py(r[c]) for c in ["CustomerId", "Surname"] + FEATURES + ["Exited"]},
              "prob": round(float(r["prob"]), 4)} for _, r in cust.iterrows()]

stats = {
    "base_rate": round(float(df["Exited"].mean()) * 100, 1), "n": int(len(df)),
    "insights": insights, "customers": cust_rows,
    "model": {"auc": round(auc, 4), "accuracy": round((tp + tn) / len(y), 4),
              "recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4),
              "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "holdout": int(len(y)),
              "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx],
              "hist_stay": [round(float(v), 4) for v in hist_s],
              "hist_leave": [round(float(v), 4) for v in hist_l]},
    "defaults": {"CreditScore": int(df.CreditScore.median()), "Age": int(df.Age.median()),
                 "Tenure": int(df.Tenure.median()), "Balance": 0, "NumOfProducts": 2,
                 "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": int(df.EstimatedSalary.median()),
                 "Geography": "France", "Gender": "Male"},
}
joblib.dump({"model": model, "background": bg, "categories": CATS}, "models/churn_model.joblib")
json.dump(stats, open("models/stats.json", "w"))
print(stats["base_rate"], stats["model"]["auc"], stats["model"]["accuracy"], stats["model"]["recall"])
