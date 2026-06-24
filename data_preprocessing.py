import pandas as pd
import numpy as np
from difflib import get_close_matches

# 1. ĐỌC DỮ LIỆU

df = pd.read_csv(r"C:\Lập trình Python\Tracker_Noisy.csv")

print("=== DỮ LIỆU BAN ĐẦU ===")
print(df)

# 2. XỬ LÝ GIÁ TRỊ LỖI

# credit_score không thể âm hoặc quá thấp
df.loc[df["credit_score"] < 300, "credit_score"] = np.nan

# 3. ĐIỀN KHUYẾT DỮ LIỆU

df["monthly_income"] = df["monthly_income"].fillna(
    df["monthly_income"].mean()
)

df["credit_score"] = df["credit_score"].fillna(
    df["credit_score"].mean()
)

df["debt_to_income_ratio"] = df["debt_to_income_ratio"].fillna(
    df["debt_to_income_ratio"].mean()
)

print("\n=== SAU IMPUTATION ===")
print(df)

# 4. SỬA LỖI CHÍNH TẢ

dictionary = ["Hanoi"]

def correct_spelling(text):

    if pd.isna(text):
        return text

    match = get_close_matches(
        str(text),
        dictionary,
        n=1,
        cutoff=0.5
    )

    return match[0] if match else text

df["city"] = df["city"].apply(correct_spelling)

print("\n=== SAU SỬA CHÍNH TẢ ===")
print(df)

# 5. XỬ LÝ OUTLIER BẰNG IQR

columns = [
    "monthly_income",
    "loan_payment",
    "investment_amount",
    "actual_savings"
]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"\n--- {col} ---")
    print("Q1 =", round(Q1, 2))
    print("Q3 =", round(Q3, 2))
    print("IQR =", round(IQR, 2))

    # Gọt phẳng Outlier
    df[col] = np.where(
        df[col] < lower,
        lower,
        np.where(
            df[col] > upper,
            upper,
            df[col]
        )
    )

print("\n=== SAU XỬ LÝ IQR ===")
print(df)

# 6. LƯU DỮ LIỆU ĐÃ LÀM SẠCH

df.to_csv(
    r"C:\Lập trình Python\Tracker_Cleaned.csv",
    index=False
)

print("\nĐã lưu file Tracker_Cleaned.csv")
