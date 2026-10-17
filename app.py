"""ChurnAI dashboard: Flask backend. Run `python train.py` once, then `python app.py`."""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
log = logging.getLogger("ChurnAI")
logging.basicConfig(level=logging.INFO)

art = joblib.load(BASE / "models" / "churn_model.joblib")
MODEL, BG, CATS = art["model"], art["background"], art["categories"]
STATS = json.loads((BASE / "models" / "stats.json").read_text())
FEATURES = list(BG.columns)
LABELS = {"CreditScore": "Credit score", "Geography": "Country", "Gender": "Gender", "Age": "Age",
          "Tenure": "Tenure", "Balance": "Balance", "NumOfProducts": "Products held",
          "HasCrCard": "Credit card", "IsActiveMember": "Activity", "EstimatedSalary": "Salary"}

app = Flask(__name__)


def encode(df):
    df = df[FEATURES].copy()
    for c, cats in CATS.items():
        df[c] = df[c].astype("category").cat.set_categories(cats)
    return df


def score(rows):
    return MODEL.predict_proba(encode(pd.DataFrame(rows)))[:, 1]


def clean(d):
    out = {
        "CreditScore": float(d["CreditScore"]), "Age": float(d["Age"]), "Tenure": float(d["Tenure"]),
        "Balance": float(d["Balance"]), "NumOfProducts": int(d["NumOfProducts"]),
        "HasCrCard": int(d["HasCrCard"]), "IsActiveMember": int(d["IsActiveMember"]),
        "EstimatedSalary": float(d["EstimatedSalary"]),
        "Geography": str(d["Geography"]), "Gender": str(d["Gender"]),
    }
    if out["Geography"] not in CATS["Geography"] or out["Gender"] not in CATS["Gender"]:
        raise ValueError("Unknown country or gender")
    return out


def explain(x, p):
    """Effect of each trait: score minus the score when that trait is swapped for typical customers."""
    out = []
    for f in FEATURES:
        rows = [{**x, f: v} for v in BG[f].tolist()]
        out.append({"feature": f, "label": LABELS[f], "effect": float(p - score(rows).mean())})
    out.sort(key=lambda r: -abs(r["effect"]))
    return out


def playbook(x, p):
    options = []
    if not x["IsActiveMember"]:
        options.append(("Re-engage as an active member", {"IsActiveMember": 1}))
    if x["NumOfProducts"] != 2:
        options.append(("Move to two products", {"NumOfProducts": 2}))
    if not x["HasCrCard"]:
        options.append(("Offer a credit card", {"HasCrCard": 1}))
    if x["NumOfProducts"] == 2 and x["IsActiveMember"]:
        options.append(("Add a third product", {"NumOfProducts": 3}))
    if x["Balance"] > 0:
        options.append(("Offer a balance-linked savings deal (balance -25%)", {"Balance": x["Balance"] * 0.75}))
    res = []
    for title, change in options:
        new = float(score([{**x, **change}])[0])
        res.append({"title": title, "change": change, "new_prob": new, "delta": new - p})
    res.sort(key=lambda r: r["delta"])
    return [r for r in res if r["delta"] < -0.005][:4]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    return jsonify({**STATS, "categories": CATS})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        x = clean(request.get_json(force=True))
        p = float(score([x])[0])
        return jsonify({"prob": p, "explain": explain(x, p), "playbook": playbook(x, p)})
    except (KeyError, ValueError, TypeError) as e:
        log.warning("Bad input: %s", e)
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True, "rows_trained": STATS["n"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
