p95 latency budget: 50 ms. Winner: **logreg**.

| Model | F1 | ROC AUC | p95 ms | Size KB | contract/month-to-month | contract/one-year | contract/two-year | internet_service/dsl | internet_service/fiber | internet_service/none | Run ID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **logreg** (champion) | 0.5948 | 0.794 | 5.112 | 3.5635 | 0.6354 | 0.4444 | 0.3077 | 0.697 | 0.597 | 0.375 | `9d2f5183c29c45d0af48a599ee40b861` |
| random_forest | 0.5478 | 0.7281 | 30.4334 | 3523.7676 | 0.6054 | 0.3529 | 0.1818 | 0.6571 | 0.5758 | 0.1429 | `3fb78ef98b9b45a8aec2dc992532ca1a` |
| hist_gb | 0.5 | 0.731 | 9.6685 | 276.6963 | 0.5517 | 0.4 | 0.0 | 0.64 | 0.5138 | 0.0952 | `16540aca517149c692dd3e9d1ad1ae04` |
