import pandas as pd
import numpy as np
import os

# Đọc file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Tracker_Noisy.csv")

df_tracker = pd.read_csv(csv_path)

# 1. Thay giá trị vô lý
# credit_score = -999 -> NaN
df_tracker.loc[
    df_tracker["credit_score"] < 0,
    "credit_score"
] = np.nan

# 2. Xử lý giá trị thiếu

cols = [
    "monthly_income",
    "credit_score",
    "debt_to_income_ratio",
    "financial_stress_level"
]

# Cột số
for col in [
    "monthly_income",
    "credit_score",
    "debt_to_income_ratio"
]:
    df_tracker[col] = df_tracker[col].fillna(
        df_tracker[col].median()
    )

# Cột chữ
df_tracker["financial_stress_level"] = (
    df_tracker["financial_stress_level"]
    .fillna(
        df_tracker["financial_stress_level"].mode()[0]
    )
)


# 3. Lưu file sạch

output_path = os.path.join(BASE_DIR, "Tracker_Clean.csv")

df_tracker.to_csv(output_path, index=False)

print("Đã làm sạch Tracker_Noisy.csv")
print("File lưu tại:", output_path)
