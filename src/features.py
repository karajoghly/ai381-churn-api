"""Single source of truth for the model's input contract.

Both training (src/train/train.py) and serving (src/app/main.py) import from here,
so a renamed column (Lab 1, Twist 1: monthly_charges -> monthly_fee) can never
silently drift between the two again.
"""

TARGET = "churn"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = ["tenure_months", "monthly_fee", "support_calls"]
CATEGORICAL_FEATURES = ["contract", "internet_service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CONTRACT_VALUES = ("month-to-month", "one-year", "two-year")
INTERNET_SERVICE_VALUES = ("fiber", "dsl", "none")

RANDOM_STATE = 42
