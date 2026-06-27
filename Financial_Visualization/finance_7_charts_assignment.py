
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("python_finance_outputs")
out_dir.mkdir(exist_ok=True)

income = pd.read_csv("Income_CLEANED.csv")
expenses = pd.read_csv("Expenses_CLEANED.csv")
budget = pd.read_csv("Budget_CLEANED.csv")
tracker = pd.read_csv("Tracker_Clean.csv")
extended = pd.read_csv("8000_Extended_Clean")

for df, col in [(income, "date_time"), (expenses, "date_time"), (budget, "date"), (tracker, "date"), (extended, "Date")]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# 1. Cán cân thanh khoản tiền mặt ngắn hạn
total_income = income["amount"].sum()
total_expenses = expenses["amount"].sum()
net_cash_flow = total_income - total_expenses
consumption_ratio = total_expenses / total_income * 100

pd.DataFrame({
    "metric": ["Tổng thu nhập", "Tổng chi tiêu", "Dòng tiền ròng", "Tỷ lệ tiêu dùng so với thu nhập (%)"],
    "value": [total_income, total_expenses, net_cash_flow, consumption_ratio]
}).to_csv(out_dir / "01_can_can_thanh_khoan.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(["Tổng thu nhập", "Tổng chi tiêu", "Dòng tiền ròng"], [total_income, total_expenses, net_cash_flow])
plt.title("1. Cán cân thanh khoản tiền mặt ngắn hạn")
plt.ylabel("Số tiền")
plt.tight_layout()
plt.savefig(out_dir / "01_column_chart_can_can_thanh_khoan.png", dpi=200)
plt.close()

# 2. Độ lệch ròng ngân sách thực tế
def normalize_cat(s):
    return str(s).strip().lower().replace("restuarant", "restaurant").replace("coffe", "cafe")

expenses["category_norm"] = expenses["category"].map(normalize_cat)
budget["category_norm"] = budget["category"].map(normalize_cat)

actual_by_cat = expenses.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "actual_spending"})
budget_by_cat = budget.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "budget_amount"})

budget_compare = pd.merge(budget_by_cat, actual_by_cat, on="category_norm", how="outer").fillna(0)
budget_compare["variance"] = budget_compare["budget_amount"] - budget_compare["actual_spending"]
budget_compare["status"] = np.where(budget_compare["variance"] < 0, "Vượt ngân sách", "Chi dưới/dư ngân sách")
budget_compare = budget_compare.sort_values("variance")
budget_compare.to_csv(out_dir / "02_do_lech_ngan_sach.csv", index=False)

plt.figure(figsize=(10, max(5, len(budget_compare) * 0.35)))
plt.barh(budget_compare["category_norm"], budget_compare["variance"])
plt.axvline(0)
plt.title("2. Độ lệch ròng ngân sách thực tế theo danh mục")
plt.xlabel("Ngân sách - Chi thực tế")
plt.ylabel("Danh mục")
plt.tight_layout()
plt.savefig(out_dir / "02_diverging_bar_chart_ngan_sach.png", dpi=200)
plt.close()

# 3. Xu hướng tiêu dùng tích lũy dài hạn
if "TransactionType" in extended.columns:
    expense_ext = extended[extended["TransactionType"].astype(str).str.lower().str.contains("expense|debit|spent|payment", na=False)].copy()
    if expense_ext.empty:
        expense_ext = extended.copy()
else:
    expense_ext = extended.copy()

long_term_cat = expense_ext.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
long_term_cat["share_percent"] = long_term_cat["Amount"] / long_term_cat["Amount"].sum() * 100
long_term_cat.to_csv(out_dir / "03_tich_luy_dai_han_theo_danh_muc.csv", index=False)

top_long = long_term_cat.head(15).sort_values("Amount")
plt.figure(figsize=(10, 6))
plt.barh(top_long["Category"], top_long["Amount"])
plt.title("3. Xu hướng tiêu dùng tích lũy dài hạn - Top danh mục")
plt.xlabel("Tổng chi tiêu tích lũy")
plt.ylabel("Danh mục")
plt.tight_layout()
plt.savefig(out_dir / "03_bar_chart_tieu_dung_tich_luy.png", dpi=200)
plt.close()

# 4. Biên độ chi tiêu linh hoạt theo đòn bẩy nợ và stress
stress = tracker["financial_stress_level"].astype(str).str.lower()
stress_high = stress.str.contains("high|cao", na=False)
dti_threshold = tracker["debt_to_income_ratio"].quantile(0.75)

tracker["risk_group"] = np.where(
    (tracker["debt_to_income_ratio"] >= dti_threshold) & stress_high,
    "DTI cao + stress cao",
    "Nhóm bình thường"
)

box_data = [
    tracker.loc[tracker["risk_group"] == "Nhóm bình thường", "discretionary_spending"].dropna(),
    tracker.loc[tracker["risk_group"] == "DTI cao + stress cao", "discretionary_spending"].dropna()
]

tracker.groupby("risk_group", as_index=False).agg(
    count=("user_id", "count"),
    avg_dti=("debt_to_income_ratio", "mean"),
    avg_discretionary_spending=("discretionary_spending", "mean"),
    median_discretionary_spending=("discretionary_spending", "median"),
    avg_monthly_expense=("monthly_expense_total", "mean")
).to_csv(out_dir / "04_so_sanh_nhom_rui_ro.csv", index=False)

plt.figure(figsize=(8, 5))
plt.boxplot(box_data, labels=["Nhóm bình thường", "DTI cao + stress cao"])
plt.title("4. Biên độ chi tiêu linh hoạt theo đòn bẩy nợ và stress")
plt.ylabel("Chi tiêu linh hoạt")
plt.tight_layout()
plt.savefig(out_dir / "04_boxplot_no_stress.png", dpi=200)
plt.close()

# 5. Ma trận tương quan tuyến tính đa biến
corr_cols = [
    "monthly_income", "monthly_expense_total", "savings_rate", "budget_goal",
    "credit_score", "debt_to_income_ratio", "loan_payment",
    "investment_amount", "emergency_fund", "transaction_count",
    "discretionary_spending", "essential_spending", "rent_or_mortgage",
    "actual_savings", "savings_goal_met"
]
corr = tracker[corr_cols].corr(numeric_only=True)
corr.to_csv(out_dir / "05_ma_tran_tuong_quan.csv")

plt.figure(figsize=(11, 9))
im = plt.imshow(corr, aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("5. Ma trận tương quan tuyến tính đa biến")
plt.tight_layout()
plt.savefig(out_dir / "05_heatmap_tuong_quan.png", dpi=200)
plt.close()

# 6. Mật độ phân phối điểm tín nhiệm & rủi ro dưới chuẩn
credit = tracker["credit_score"].dropna()
safe_threshold = 670
below_safe_count = int((credit < safe_threshold).sum())
below_safe_rate = below_safe_count / len(credit) * 100
credit_mean = credit.mean()

pd.DataFrame({
    "metric": ["Điểm tín dụng trung bình", "Ngưỡng an toàn giả định", "Số người dưới ngưỡng", "Tỷ lệ dưới ngưỡng (%)"],
    "value": [credit_mean, safe_threshold, below_safe_count, below_safe_rate]
}).to_csv(out_dir / "06_phan_phoi_diem_tin_nhiem.csv", index=False)

plt.figure(figsize=(9, 5))
credit.plot(kind="kde")
plt.axvline(safe_threshold, linestyle="--")
plt.axvline(credit_mean, linestyle=":")
plt.title("6. Mật độ phân phối điểm tín nhiệm & rủi ro dưới chuẩn")
plt.xlabel("Credit score")
plt.ylabel("Mật độ")
plt.tight_layout()
plt.savefig(out_dir / "06_kde_credit_score.png", dpi=200)
plt.close()

# 7. Tương quan thu nhập với bẫy lạm phát lối sống
tracker.groupby("savings_goal_met", as_index=False).agg(
    count=("user_id", "count"),
    avg_income=("monthly_income", "mean"),
    avg_expense=("monthly_expense_total", "mean"),
    avg_savings_rate=("savings_rate", "mean")
).to_csv(out_dir / "07_thu_nhap_va_bay_lam_phat_loi_song.csv", index=False)

plt.figure(figsize=(9, 6))
for val in sorted(tracker["savings_goal_met"].dropna().unique()):
    subset = tracker[tracker["savings_goal_met"] == val]
    label = "Đạt mục tiêu tiết kiệm" if val == 1 else "Không đạt mục tiêu tiết kiệm"
    plt.scatter(subset["monthly_income"], subset["monthly_expense_total"], alpha=0.6, label=label)
plt.title("7. Tương quan thu nhập với bẫy lạm phát lối sống")
plt.xlabel("Thu nhập hàng tháng")
plt.ylabel("Tổng chi tiêu hàng tháng")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "07_scatter_income_vs_expense.png", dpi=200)
plt.close()

print("Hoàn thành. Kết quả nằm trong thư mục:", out_dir)
