# Model selection

**Champion: Logistic Regression with balanced class weights** (`config/train.yaml`),
registered as `churn-model@champion`.

It has the best F1 of the three candidates (0.595), the best F1 on 5 of the 6 customer
slices, the lowest latency and a model 1,000 times smaller than the random forest.
The exact numbers, latencies and MLflow run IDs of the last run are in
[`reports/candidates.md`](reports/candidates.md); each candidate is a run in the
`churn` experiment in MLflow.

## How the candidates were compared

- **Same data, same split.** `data/churn_v2.csv` (SHA-256 logged on every run), one stratified
  80/20 split with `random_state=42`, so every model is scored on the same 300 customers.
- **F1, not accuracy.** Only 31% of customers churn. A model that always says "no churn" is 69%
  accurate and useless; its F1 is 0. F1 balances finding churners (recall) against false alarms
  (precision), which is what a retention team acts on.
- **F1 per slice.** A model can look fine overall and fail one customer group, so F1 is also
  computed per `contract` type and per `internet_service` type.
- **p95 latency of one prediction**, the way `/predict` is called, against a **50 ms budget**.
- **Model size**, because it is shipped inside the container image.

## Results (test set, 300 customers)

| Model | F1 | ROC AUC | p95 latency | Size |
|---|---|---|---|---|
| **Logistic Regression (balanced)** | **0.595** | **0.794** | **~4 ms** | **3.6 KB** |
| Random Forest (200 trees, depth 8, balanced) | 0.548 | 0.728 | ~19 ms | 3,524 KB |
| Hist Gradient Boosting (200 iters, depth 4) | 0.500 | 0.731 | ~6 ms | 277 KB |

F1 by slice:

| Slice | Logistic Regression | Random Forest | Hist GB |
|---|---|---|---|
| contract = month-to-month | **0.635** | 0.605 | 0.552 |
| contract = one-year | **0.444** | 0.353 | 0.400 |
| contract = two-year | **0.308** | 0.182 | 0.000 |
| internet_service = dsl | **0.697** | 0.657 | 0.640 |
| internet_service = fiber | **0.597** | 0.576 | 0.514 |
| internet_service = none | **0.375** | 0.143 | 0.095 |

Latency is measured on the machine that trains, so it varies a little run to run; the
other numbers are deterministic.

## Why Logistic Regression

1. **Best quality where it matters.** Highest F1 overall and on every slice. The tree models
   score F1 = 0 or close to it on two-year contracts and customers with no internet: they
   almost never flag those small groups, which the overall number hides.
2. **The data is close to linear.** 1,500 rows and 5 features; churn rises steadily with fewer
   months, higher fees and more support calls. The trees have more capacity than the data can
   use and overfit (lower ROC AUC on the test set than the linear model).
3. **Cheapest to run.** ~4 ms per prediction and 3.6 KB, so a cold start after scale-to-zero
   and the image size are unaffected.
4. **Explainable.** Each coefficient says how a feature moves the churn odds, which helps
   explain a prediction to the business.

## Trade-offs and risks

- **Precision 0.50.** Balanced class weights trade precision for recall (0.73): about half the
  customers flagged will not churn. That is acceptable if a retention offer is cheap; if it is
  expensive, raise the decision threshold instead of changing the model.
- **Accuracy equals the majority baseline (0.687).** Expected with balanced weights; it is why
  the gate and the tests use F1, not accuracy.
- **Small slices.** The two-year and no-internet slices have only a few churners in the test set,
  so their F1 is noisy. A larger test set or cross-validation would be the next step.

## The promotion gate (`config/train.yaml`)

A retrained challenger replaces `@champion` only if **F1 ≥ champion F1 + `min_delta` (0.0)**
**and p95 latency ≤ 50 ms**. A tie is allowed on purpose: retraining the same code on the same
data reproduces the same model, and that must still be deployable. Raise `min_delta` to demand
a real improvement.
