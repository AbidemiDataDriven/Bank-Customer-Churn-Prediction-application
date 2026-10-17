# ChurnAI: Bank Customer Churn Dashboard

An interactive retention dashboard for a retail bank. It predicts how likely each customer is to leave, explains the score, and suggests the actions that would lower the risk the most.

Built with **Python, Flask, scikit-learn and vanilla JavaScript**. There is no front-end framework, no build step and no charting library.

<!--
Add screenshots here once you have run the app, for example:
![Simulator](docs/simulator.png)
-->

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Quick start](#quick-start)
4. [Project structure](#project-structure)
5. [The dataset](#the-dataset)
6. [The model](#the-model)
7. [Model performance](#model-performance)
8. [What the data says about churn](#what-the-data-says-about-churn)
9. [Dashboard guide](#dashboard-guide)
10. [API reference](#api-reference)
11. [Configuration and customisation](#configuration-and-customisation)
12. [Deployment](#deployment)
13. [Limitations and responsible use](#limitations-and-responsible-use)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)
16. [Contributing](#contributing)
17. [License](#license)

---

## Features

| Tab | What it does |
|---|---|
| **Simulator** | Sliders, switches and dropdowns for every customer trait. The risk dial updates live as you change anything. There is no form to submit. |
| **Why this score** | Diverging bars show which traits push a customer toward leaving and which keep them, compared with a typical customer. |
| **Retention playbook** | Candidate actions ranked by how much each lowers the predicted risk. One click applies an action to the simulator. |
| **Insights** | Churn rate by age, products held, country, activity, credit score, tenure and gender, each compared with the overall average. |
| **Customers** | 400 real customers from the dataset that the model did not train on, scored and searchable by surname or ID. Click a row to load that customer into the simulator. |
| **Model check** | AUC, accuracy, recall and precision, a leaver/stayer separation chart, an ROC curve and a confusion matrix. |

Other details:

- Light and dark themes follow the operating system setting.
- Responsive layout that works on phones.
- Input validation on the server: unknown countries or genders and malformed numbers return a clean `400` error.
- A `/health` endpoint for uptime checks.

---

## How it works

```
data/train.csv
      |
      v
  train.py ----> models/churn_model.joblib   (model, background sample, category lists)
      |    \---> models/stats.json           (insights, metrics, customer list, defaults)
      v
   app.py  (Flask)
      |-- GET  /              -> templates/index.html (single-page dashboard)
      |-- GET  /api/stats     -> precomputed statistics for the charts and customer list
      |-- POST /api/predict   -> probability + explanation + playbook for one customer
      '-- GET  /health
```

1. **Training** (`train.py`) reads the CSV, trains a gradient-boosted classifier on 80% of the rows, evaluates it on the remaining 20%, and precomputes everything the dashboard needs.
2. **Serving** (`app.py`) loads the saved model and statistics once at startup.
3. **Live scoring.** When you move a control, the page waits about 110 ms, sends the profile to `/api/predict`, and renders the response. Out-of-order responses are discarded so the dial never shows a stale result.

### How the explanation is computed

For each trait, the app swaps that trait's value for values taken from a background sample of 300 real customers, scores each variant, and averages the results. The **effect** of the trait is the customer's actual score minus that average.

- A positive effect means the trait raises this customer's risk relative to a typical customer.
- A negative effect means it lowers the risk.
- Effects are in percentage points of churn probability.

This is a simple, model-agnostic attribution method. It is not SHAP, and effects will not sum exactly to the difference between the score and the average, because the model has interactions between traits.

### How the playbook is computed

The app builds a short list of candidate actions that apply to the customer's current profile, re-scores the customer with each change applied, and keeps the ones that lower the risk by more than 0.5 percentage points (up to four, best first).

| Candidate action | Change tested |
|---|---|
| Re-engage as an active member | `IsActiveMember = 1` (if currently inactive) |
| Move to two products | `NumOfProducts = 2` (if currently not 2) |
| Offer a credit card | `HasCrCard = 1` (if the customer has none) |
| Add a third product | `NumOfProducts = 3` (if the customer has two products and is active) |
| Offer a balance-linked savings deal | `Balance` reduced by 25% (if balance is above zero) |

> The playbook shows what the **model** predicts would happen if a trait changed. It does not prove that the real-world action would cause that change. See [Limitations](#limitations-and-responsible-use).

---

## Quick start

### Requirements

- Python 3.9 or newer
- About 300 MB of free disk space for dependencies

### Install and run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (takes under a minute on a typical laptop)
python train.py

# 5. Start the dashboard
python app.py
```

Open **http://localhost:5000**.

`train.py` prints the average churn rate, AUC, accuracy and recall when it finishes. Step 4 is optional if the `models/` folder already contains `churn_model.joblib` and `stats.json`.

> **scikit-learn version.** The saved `.joblib` file is only guaranteed to load with the scikit-learn version it was trained with. If you see a version warning or error, rerun `python train.py` in your own environment.

---

## Project structure

```
.
├── app.py                  # Flask app: routes, validation, explanation, playbook
├── train.py                # Model training and statistics precomputation
├── requirements.txt        # flask, pandas, scikit-learn, joblib
├── data/
│   └── train.csv           # Customer data (165,034 rows)
├── models/
│   ├── churn_model.joblib  # Trained model, background sample, category lists
│   └── stats.json          # Insights, metrics, 400 scored customers, defaults
└── templates/
    └── index.html          # Entire front end: HTML, CSS and JavaScript
```

---

## The dataset

`data/train.csv` contains **165,034 customers** and **14 columns**, with no missing values. It follows the layout of the Kaggle "Bank Customer Churn" playground data.

| Column | Type | Description |
|---|---|---|
| `id` | integer | Row identifier (not used by the model) |
| `CustomerId` | integer | Customer identifier (not used by the model) |
| `Surname` | text | Customer surname (not used by the model) |
| `CreditScore` | number | Credit score |
| `Geography` | category | France, Germany or Spain |
| `Gender` | category | Male or Female |
| `Age` | number | Age in years |
| `Tenure` | number | Years as a customer |
| `Balance` | number | Account balance in euros |
| `NumOfProducts` | integer | Number of bank products held |
| `HasCrCard` | 0 / 1 | Has a credit card |
| `IsActiveMember` | 0 / 1 | Active member |
| `EstimatedSalary` | number | Estimated salary in euros |
| `Exited` | 0 / 1 | **Target:** 1 if the customer left |

The overall churn rate is **21.2%**.

To use your own data, keep the same column names (see [Configuration](#configuration-and-customisation)).

---

## The model

| Setting | Value |
|---|---|
| Algorithm | `HistGradientBoostingClassifier` (scikit-learn) |
| Boosting iterations | 300 |
| Learning rate | 0.08 |
| Max leaf nodes | 31 |
| Categorical handling | Native (`Geography`, `Gender`) |
| Input features | 10: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` |
| Excluded | `id`, `CustomerId`, `Surname` |
| Train / test split | 80% / 20%, stratified on `Exited`, `random_state=42` |

Design choices:

- **Gradient boosting** handles the strong non-linear patterns in this data (churn is far higher at certain ages and product counts) without manual feature engineering.
- **Identifiers are excluded** so the model cannot memorise individual customers and the simulator inputs match what a relationship manager would actually know.
- **One model serves everything.** The customer list, metrics and live simulator all use the model trained on the 80% split, and the 400 listed customers come from the held-out 20%, so the scores you see for them are honest out-of-sample predictions.

---

## Model performance

Measured on **33,007 held-out customers** the model never saw during training, using a 50% decision threshold.

| Metric | Value | Meaning |
|---|---|---|
| **AUC** | 0.890 | How well the model ranks leavers above stayers (1.0 is perfect, 0.5 is random) |
| **Accuracy** | 86.6% | Share of all calls that were right |
| **Precision** | 74.7% | When the model flags a customer as likely to leave, how often it is right |
| **Recall** | 55.2% | Share of actual leavers the model catches |

### Confusion matrix

|  | Predicted stay | Predicted leave |
|---|---:|---:|
| **Actually stayed** | 24,717 (correct) | 1,306 (false alarm) |
| **Actually left** | 3,128 (missed) | 3,856 (correct) |

### Reading these numbers

The model is good at **ranking** customers by risk (AUC 0.89) but conservative at the 50% cut-off, so it misses about 45% of leavers. If missing a leaver costs more than contacting a stayer, use a lower threshold or work down the list from the highest risk score instead of relying on a yes/no flag. Lowering the threshold raises recall and lowers precision.

---

## What the data says about churn

Average churn across all customers is **21.2%**. The largest differences, from the full dataset:

| Group | Churn rate |
|---|---:|
| Age 18-29 | 8.4% |
| Age 30-39 | 10.8% |
| Age 40-49 | 34.8% |
| **Age 50-59** | **60.9%** |
| Age 60+ | 34.9% |
| 1 product | 34.7% |
| **2 products** | **6.0%** |
| 3 products | 88.3% |
| 4 products | 87.6% |
| France | 16.5% |
| **Germany** | **37.9%** |
| Spain | 17.2% |
| Active member | 12.5% |
| Inactive member | 29.7% |
| Female | 28.0% |
| Male | 15.9% |

Key patterns:

- Customers with **two products** are by far the most loyal. Those with three or four products almost all leave.
- Churn peaks for customers in their **fifties**.
- **German** customers leave at more than twice the rate of French and Spanish customers.
- **Inactive** members churn at more than twice the rate of active members.

These are correlations in historical data. They describe who left, not necessarily why.

---

## Dashboard guide

### Simulator

| Control | Range |
|---|---|
| Credit score | 300 to 850 |
| Age | 18 to 92 |
| Tenure | 0 to 10 years |
| Balance | €0 to €250,000 |
| Estimated salary | €0 to €200,000 |
| Products held | 1 to 4 |
| Country | France, Germany, Spain |
| Gender | Female, Male |
| Credit card, Active member | On / off |

The dial shows the predicted churn probability. A tick mark on the dial shows the 21.2% average, so you can see at a glance whether a customer is above or below typical.

The verdict thresholds are:

| Risk | Label |
|---|---|
| Under 20% | Likely to stay |
| 20% to 49% | Worth watching |
| 50% and above | Likely to leave |

### Customers

Loads 400 customers sampled from the held-out test split, sorted by risk. Search by surname or ID. The final column shows whether the customer actually left, so you can compare the model's score with reality.

### Model check

All charts on this tab come from the same 20% hold-out used for the metrics above.

---

## API reference

All responses are JSON unless noted.

### `GET /`

Returns the dashboard page (HTML).

### `GET /api/stats`

Returns the precomputed statistics used to draw the dashboard.

| Field | Description |
|---|---|
| `base_rate` | Overall churn rate, in percent |
| `n` | Number of training rows |
| `insights` | Churn rate and group size for each breakdown |
| `customers` | The 400 scored sample customers |
| `model` | Metrics, ROC points, score histograms, confusion matrix counts |
| `defaults` | Starting profile for the simulator |
| `categories` | Allowed values for `Geography` and `Gender` |

### `POST /api/predict`

Scores one customer.

**Request body**

```json
{
  "CreditScore": 650,
  "Age": 48,
  "Tenure": 3,
  "Balance": 120000,
  "EstimatedSalary": 90000,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "Geography": "Germany",
  "Gender": "Female"
}
```

**Example**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":650,"Age":48,"Tenure":3,"Balance":120000,"EstimatedSalary":90000,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":0,"Geography":"Germany","Gender":"Female"}'
```

**Response**

```json
{
  "prob": 0.885,
  "explain": [
    {"feature": "Age", "label": "Age", "effect": 0.283},
    {"feature": "Geography", "label": "Country", "effect": 0.228}
  ],
  "playbook": [
    {
      "title": "Move to two products",
      "change": {"NumOfProducts": 2},
      "new_prob": 0.65,
      "delta": -0.234
    }
  ]
}
```

(The lists above are shortened. The real response includes all ten traits in `explain` and up to four actions in `playbook`.)

| Field | Description |
|---|---|
| `prob` | Churn probability between 0 and 1 |
| `explain` | Every trait with its `effect`, sorted by absolute size |
| `playbook` | Up to four actions with the changed fields, the new probability and the change (`delta`) |

**Errors.** A missing field, a non-numeric value or an unknown `Geography` or `Gender` returns HTTP `400` with `{"error": "..."}`.

### `GET /health`

```json
{"status": "healthy", "model_loaded": true, "rows_trained": 165034}
```

---

## Configuration and customisation

### Use your own data

1. Replace `data/train.csv` with a file that has the same columns (see [The dataset](#the-dataset)).
2. Run `python train.py`.
3. Restart `python app.py`.

If your column names differ, update `FEATURES`, `CAT` and the field handling in `train.py`, the `clean()` function and `LABELS` in `app.py`, and the controls in `templates/index.html`.

### Tune the model

Edit the `HistGradientBoostingClassifier(...)` call in `train.py`. Common changes are `max_iter`, `learning_rate`, `max_leaf_nodes` and `l2_regularization`. Rerun training and compare the AUC and recall printed at the end.

### Change the decision thresholds

- The 50% cut-off for the confusion matrix and the recall and precision figures is set in `train.py` (`p >= 0.5`).
- The "Likely to stay / Worth watching / Likely to leave" bands are set in the `render()` function in `templates/index.html`.

### Add or change playbook actions

Edit the `playbook()` function in `app.py`. Each option is a title plus a dictionary of field changes. Add an entry and it is scored and ranked automatically.

### Change the port or host

Edit the last line of `app.py`:

```python
app.run(host="0.0.0.0", port=5000, debug=False)
```

---

## Deployment

`app.run` is Flask's development server. For production, use a WSGI server.

### Gunicorn

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

The model is loaded once per worker at import time, so memory use grows with the number of workers.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
RUN python train.py
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
```

```bash
docker build -t churnai .
docker run -p 8000:8000 churnai
```

Training inside the image build keeps the model and the scikit-learn version in sync.

### Production notes

- Put the app behind HTTPS and authentication. It handles customer data.
- The dashboard loads the IBM Plex fonts from Google Fonts. Remove the font `<link>` tags in `templates/index.html` for an offline or privacy-restricted setup; the page falls back to system fonts.

---

## Limitations and responsible use

- **Correlation, not causation.** The playbook shows how the model's prediction changes when a trait changes. Moving a real customer to two products will not necessarily make them stay. Test retention actions with a controlled pilot before acting at scale.
- **Unusual patterns in the data.** Customers with three or four products almost all churn in this dataset. That may reflect how the data was generated or how the bank sold products, so check it against your own business knowledge.
- **Recall is moderate.** At a 50% threshold the model misses about 45% of leavers. Do not treat "Likely to stay" as a guarantee.
- **Fairness.** The model uses `Gender` and `Geography` as inputs, and the data shows large churn differences across them. Before using scores to make decisions that affect customers, review whether using these traits is appropriate and lawful in your jurisdiction, and consider removing them.
- **Data drift.** Customer behaviour changes. Retrain regularly and monitor performance on recent data.
- **Explanations are approximate.** The "Why this score" bars come from a simple swap-and-compare method, not an exact attribution.
- **Privacy.** Do not commit real customer data to a public repository. If `data/train.csv` is real, add it to `.gitignore`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for `churn_model.joblib` on startup | Run `python train.py` first. |
| Warning or error about scikit-learn versions when loading the model | Retrain with `python train.py` in your current environment. |
| `ModuleNotFoundError` | Activate your virtual environment and run `pip install -r requirements.txt`. |
| Page loads but the dial shows "Could not score" | Check the terminal for a Flask error and confirm `/api/predict` responds with `curl`. |
| Port 5000 already in use | Change `port=5000` in `app.py`. On macOS, AirPlay Receiver often uses port 5000. |
| Fonts look different from the screenshots | The Google Fonts request is blocked. System fonts are used instead, which is harmless. |

---

## Roadmap

Ideas for future work:

- Calibrated probabilities and a threshold picker based on retention cost
- SHAP-based explanations
- Batch scoring: upload a CSV and download scored results
- Experiment tracking and automated retraining
- Unit tests for the API and the playbook logic
- Authentication and role-based access

---

## Contributing

Contributions are welcome.

1. Fork the repository and create a branch: `git checkout -b feature/my-change`
2. Make your change and confirm that `python train.py` and `python app.py` still run cleanly.
3. Open a pull request describing what you changed and why.

Please open an issue first for large changes.

---

## License

No license has been chosen yet. Add a `LICENSE` file (for example MIT) before publishing so others know how they may use the code, then replace this section with the license name.