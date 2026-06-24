import pandas as pd
import numpy as np
import os

# Đọc file CSV nằm cùng thư mục với file .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "8000_Extended_Noisy.csv")

df_8000 = pd.read_csv(csv_path)

# 1. Sửa lỗi chính tả Category

df_8000["Category"] = df_8000["Category"].replace({
    "Travell_err": "Travel",
    "Grocerie$": "Grocery",
    "Biills": "Bills"
})

# 2. Xử lý giá trị thiếu

# Category -> thay bằng giá trị xuất hiện nhiều nhất
df_8000["Category"] = df_8000["Category"].fillna(
    df_8000["Category"].mode()[0]
)

# Amount -> thay bằng trung vị
df_8000["Amount"] = df_8000["Amount"].fillna(
    df_8000["Amount"].median()
)

# 3. Xử lý Outlier Amount

Q1 = df_8000["Amount"].quantile(0.25)
Q3 = df_8000["Amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_amount = df_8000["Amount"].median()

df_8000.loc[
    (df_8000["Amount"] < lower_bound) |
    (df_8000["Amount"] > upper_bound),
    "Amount"
] = median_amount

# 4. Lưu file sạch

output_path = os.path.join(BASE_DIR, "8000_Extended_Clean.csv")
df_8000.to_csv(output_path, index=False)

print("Đã làm sạch dữ liệu!")
print("File lưu tại:", output_path)
