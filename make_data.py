import pandas as pd, numpy as np

rng = np.random.default_rng(0)
n = 5000
df = pd.DataFrame({
    "Date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
    "Customer_ID": rng.integers(1000, 1200, n),
    "Account_Type": rng.choice(["Current", "Savings"], n),
    "Transaction_Type": rng.choice(["Debit", "Credit"], n),
    "Amount": np.round(rng.lognormal(4, 1.1, n), 2),
    "Country": rng.choice(["CA", "US", "FR"], n, p=[.6, .3, .1]),
    "Merchant_Category": rng.choice(["Grocery", "Travel", "Utilities", "Dining"], n),
})
df.to_csv("data/tx.csv", index=False)
print(df.shape)