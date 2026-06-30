import pandas as pd
import numpy as np
import os

# 1. Đọc dữ liệu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Tracker_Noisy.csv")

df_track = pd.read_csv(csv_path)

print("========== THÔNG TIN BAN ĐẦU ==========")
print(df_track.info())
print("\nKích thước dữ liệu:", df_track.shape)

# 2. Thay giá trị vô lý
# credit_score = -999 -> NaN

df_track["credit_score"] = df_track["credit_score"].replace(-999.0, np.nan)

# 3. Điền giá trị thiếu

numeric_cols = [
    "monthly_income",
    "credit_score",
    "debt_to_income_ratio"
]

for col in numeric_cols:
    df_track[col] = df_track[col].fillna(df_track[col].median())

df_track["financial_stress_level"] = (
    df_track["financial_stress_level"]
    .fillna(df_track["financial_stress_level"].mode()[0])
)

# 4. Xử lý Outlier bằng IQR
# (Thay bằng Median)

Q1 = df_track["monthly_income"].quantile(0.25)
Q3 = df_track["monthly_income"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

median_income = df_track["monthly_income"].median()

outlier_count = (
    (df_track["monthly_income"] < lower) |
    (df_track["monthly_income"] > upper)
).sum()

print("\nSố Outlier của monthly_income:", outlier_count)

df_track.loc[
    (df_track["monthly_income"] < lower) |
    (df_track["monthly_income"] > upper),
    "monthly_income"
] = median_income

# 5. Kiểm tra sau khi clean

print("\n========== GIÁ TRỊ THIẾU ==========")
print(df_track.isnull().sum())

print("\n========== THỐNG KÊ ==========")
print(df_track.describe(include="all"))

print("\n========== THÔNG TIN ==========")
print(df_track.info())

# 6. Lưu dữ liệu sạch

output_path = os.path.join(BASE_DIR, "Tracker_Clean.csv")

df_track.to_csv(output_path, index=False)

print("\n===================================")
print("Đã làm sạch dữ liệu thành công!")
print("File được lưu tại:")
print(output_path)
print("===================================")
