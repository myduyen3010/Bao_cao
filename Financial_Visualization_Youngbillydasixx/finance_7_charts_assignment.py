import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
out = Path("Financial_Visualization_Youngbillydasixx")
out.mkdir(exist_ok=True)

income = pd.read_csv("Income_CLEANED.csv")
expenses = pd.read_csv("Expenses_CLEANED.csv")
budget = pd.read_csv("Budget_CLEANED.csv")
tracker = pd.read_csv("Tracker_CLEANED.csv")
extended = pd.read_csv("8000_Extended_Clean.csv")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

def save_chart(file_name):
    plt.tight_layout()
    plt.savefig(out / file_name, dpi=300, bbox_inches="tight")
    plt.close()

def normalize_category(value):
    return str(value).strip().lower().replace("restuarant", "restaurant").replace("coffe", "cafe")

numeric_columns = [
    "monthly_income", "monthly_expense_total", "savings_rate", "budget_goal",
    "credit_score", "debt_to_income_ratio", "loan_payment", "investment_amount",
    "emergency_fund", "transaction_count", "discretionary_spending",
    "essential_spending", "rent_or_mortgage", "actual_savings", "savings_goal_met",
]
for col in numeric_columns:
    if col in tracker.columns:
        tracker[col] = pd.to_numeric(tracker[col], errors="coerce")

# 1. CÁN CÂN THANH KHOẢN TIỀN MẶT NGẮN HẠN
total_income = income["amount"].sum()
total_expenses = expenses["amount"].sum()
net_cash_flow = total_income - total_expenses
consumption_ratio = total_expenses / total_income * 100

pd.DataFrame({
    "metric": ["Tổng thu nhập", "Tổng chi tiêu", "Dòng tiền ròng", "Tỷ lệ tiêu dùng so với thu nhập (%)"],
    "value": [total_income, total_expenses, net_cash_flow, consumption_ratio]
}).to_csv(out / "01_can_can_thanh_khoan.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 5.5))
labels = ["Tổng thu nhập", "Tổng chi tiêu", "Dòng tiền ròng"]
values = [total_income, total_expenses, net_cash_flow]
colors = ["#3ad13a", "#e73737", "#3283be"]
bars = plt.bar(labels, values, color=colors)
plt.title(" CÁN CÂN THANH KHOẢN TIỀN MẶT NGẮN HẠN", fontweight="bold")
plt.ylabel("Số tiền")
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.text(0.5, max(values) * 0.82, f"Tỷ lệ tiêu dùng: {consumption_ratio:.2f}%",
         ha="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
plt.legend(handles=[
    Patch(color="#3ad13a", label=" Thu nhập"),
    Patch(color="#e73737", label=" Chi tiêu"),
    Patch(color="#3283be", label=" Dòng tiền ròng"),
], loc="upper right")
save_chart("01_column_chart_can_can_thanh_khoan.png")

# 2. BIÊN ĐỘ LỆCH RÒNG GIỮA THỰC CHI VỚI HẠN MỨC NGÂN SÁCH
expenses["category_norm"] = expenses["category"].map(normalize_category)
budget["category_norm"] = budget["category"].map(normalize_category)
actual_spending = expenses.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "actual_spending"})
budget_amount = budget.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "budget_amount"})
budget_compare = budget_amount.merge(actual_spending, on="category_norm", how="outer").fillna(0)
budget_compare["variance"] = budget_compare["budget_amount"] - budget_compare["actual_spending"]
budget_compare["status"] = np.where(budget_compare["variance"] < 0, "Vượt ngân sách", "Chi dưới ngân sách")
budget_compare = budget_compare.sort_values("variance")
budget_compare.to_csv(out / "02_do_lech_ngan_sach.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, max(7, len(budget_compare) * 0.33)))
plt.barh(budget_compare["category_norm"], budget_compare["variance"],
         color=np.where(budget_compare["variance"] >= 0, "#3ad13a", "#e73737"))
plt.axvline(0, color="black", linewidth=1)
plt.title(" BIÊN ĐỘ LỆCH RÒNG GIỮA THỰC CHI VỚI HẠN MỨC NGÂN SÁCH", fontweight="bold")
plt.xlabel("Ngân sách - Thực chi")
plt.ylabel("Danh mục")
plt.legend(handles=[
    Patch(color="#3ad13a", label=" Chi dưới ngân sách"),
    Patch(color="#e73737", label=" Vượt ngân sách"),
], loc="lower right")
save_chart("02_diverging_bar_chart_ngan_sach.png")

# 3. DANH MỤC LŨY KẾ TIÊU TỐN TIỀN MẶT DÀI HẠN
extended_expense = extended.copy()
if "TransactionType" in extended_expense.columns:
    temp = extended_expense[
        extended_expense["TransactionType"].astype(str).str.lower().str.contains("expense|debit|spent|payment", na=False)
    ]
    if not temp.empty:
        extended_expense = temp

long_term = extended_expense.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
long_term["share_percent"] = long_term["Amount"] / long_term["Amount"].sum() * 100
long_term.to_csv(out / "03_tich_luy_dai_han_theo_danh_muc.csv", index=False, encoding="utf-8-sig")

top_long_term = long_term.head(10)
gradient_colors = [plt.cm.Blues(0.9 - i * (0.45 / max(len(top_long_term) - 1, 1))) for i in range(len(top_long_term))]
plt.figure(figsize=(10, 6))
bars = plt.barh(top_long_term["Category"], top_long_term["Amount"], color=gradient_colors)
plt.gca().invert_yaxis()
plt.title(" DANH MỤC LŨY KẾ TIÊU TỐN TIỀN MẶT DÀI HẠN", fontweight="bold")
plt.xlabel("Tổng chi tiêu tích lũy")
plt.ylabel("Danh mục")
for bar, value, percent in zip(bars, top_long_term["Amount"], top_long_term["share_percent"]):
    plt.text(value, bar.get_y() + bar.get_height() / 2, f" {value / 1e6:.2f} ({percent:.1f}%)",
             va="center", fontsize=9)
save_chart("03_bar_chart_tieu_dung_tich_luy.png")

# 4. BIÊN ĐỘ CHI TIÊU THEO BIẾN ĐỊNH TÍNH ÁP LỰC VÀ NỢ
df4 = tracker[["financial_stress_level", "debt_to_income_ratio", "discretionary_spending"]].copy()
df4 = df4.replace([np.inf, -np.inf], np.nan).dropna()
df4["financial_stress_level"] = df4["financial_stress_level"].astype(str).str.strip().str.capitalize()

stress_order = ["Low", "High", "Medium"]
df4 = df4[df4["financial_stress_level"].isin(stress_order)].copy()
debt_threshold = 0.35
df4["debt_structure"] = np.where(df4["debt_to_income_ratio"] > debt_threshold, "Nguy hiểm (>35%)", "An toàn (<=35%)")

summary_4 = df4.groupby(["financial_stress_level", "debt_structure"]).agg(
    so_luong=("discretionary_spending", "count"),
    low_q1=("discretionary_spending", lambda x: x.quantile(0.25)),
    trung_binh=("discretionary_spending", "mean"),
    trung_vi=("discretionary_spending", "median"),
    high_q3=("discretionary_spending", lambda x: x.quantile(0.75)),
    min_value=("discretionary_spending", "min"),
    max_value=("discretionary_spending", "max"),
).reset_index()
summary_4.to_csv(out / "04_so_sanh_nhom_rui_ro.csv", index=False, encoding="utf-8-sig")

red = "#d62728"
blue = "#1f77b4"
positions, box_data, box_colors = [], [], []
offset = 0.18
for i, stress in enumerate(stress_order, start=1):
    dangerous_data = df4[(df4["financial_stress_level"] == stress) & (df4["debt_structure"] == "Nguy hiểm (>35%)")]["discretionary_spending"].dropna()
    safe_data = df4[(df4["financial_stress_level"] == stress) & (df4["debt_structure"] == "An toàn (<=35%)")]["discretionary_spending"].dropna()
    box_data.extend([dangerous_data, safe_data])
    positions.extend([i - offset, i + offset])
    box_colors.extend([red, blue])

plt.figure(figsize=(12, 7))
box = plt.boxplot(
    box_data, positions=positions, widths=0.32, patch_artist=True, showfliers=True,
    medianprops=dict(color="#4d4d4d", linewidth=1.7),
    whiskerprops=dict(color="#4d4d4d", linewidth=1.2),
    capprops=dict(color="#4d4d4d", linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#4d4d4d", markersize=6)
)
for patch, color in zip(box["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.92)
    patch.set_edgecolor("#4d4d4d")
for i, median in enumerate(box["medians"]):
    x_coords = median.get_xdata()
    y_coords = median.get_ydata()
    x_pos = x_coords.mean()
    y_pos = y_coords[0] 
    plt.text(
        x_pos, 
        y_pos + 12,                  
        f"{int(y_pos)}",            
        ha="center", 
        va="bottom", 
        fontsize=10, 
        fontweight="bold", 
        color="#333333"             
    )
plt.title(" BIÊN ĐỘ CHI TIÊU THEO BIẾN ĐỊNH TÍNH ÁP LỰC VÀ NỢ", fontsize=16, fontweight="bold")
plt.xlabel("Mức độ Stress (Financial Stress Level)", fontsize=14)
plt.ylabel("Chi tiêu linh hoạt (Discretionary Spending)", fontsize=14)
plt.xticks([1, 2, 3], stress_order, fontsize=13)
plt.grid(axis="y", color="#cccccc", linewidth=1, alpha=0.9)
plt.legend(handles=[
    Patch(facecolor=red, edgecolor="#4d4d4d", label="Nguy hiểm (>35%)"),
    Patch(facecolor=blue, edgecolor="#4d4d4d", label="An toàn (<=35%)"),
], title="Cấu trúc nợ", loc="lower right", frameon=True, fontsize=12, title_fontsize=13)
save_chart("04_boxplot_no_stress.png")

# 5. MA TRẬN TƯƠNG QUAN TUYẾN TÍNH ĐA BIẾN
corr_columns = [
    "monthly_income", "monthly_expense_total", "savings_rate", "budget_goal", "credit_score",
    "debt_to_income_ratio", "loan_payment", "investment_amount", "emergency_fund",
    "transaction_count", "discretionary_spending", "essential_spending",
    "rent_or_mortgage", "actual_savings", "savings_goal_met",
]
corr = tracker[corr_columns].corr(numeric_only=True)
corr.to_csv(out / "05_ma_tran_tuong_quan.csv", encoding="utf-8-sig")

plt.figure(figsize=(12, 10))
image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(image, label="Hệ số tương quan")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title(" MA TRẬN TƯƠNG QUAN TUYẾN TÍNH ĐA BIẾN", fontweight="bold")
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        value = corr.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                 color="white" if abs(value) > 0.55 else "black")
save_chart("05_heatmap_tuong_quan.png")

# 6. MẬT ĐỘ PHÂN PHỐI ĐIỂM TÍN NHIỆM TRONG QUẦN THỂ
credit = pd.to_numeric(tracker["credit_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
subprime_threshold = 600
system_mean = credit.mean()
below_count = int((credit < subprime_threshold).sum())
below_rate = below_count / len(credit) * 100

pd.DataFrame({
    "metric": ["Điểm tín dụng trung bình hệ thống", "Ngưỡng tín dụng dưới chuẩn", "Số người dưới ngưỡng", "Tỷ lệ dưới ngưỡng (%)"],
    "value": [system_mean, subprime_threshold, below_count, below_rate]
}).to_csv(out / "06_phan_phoi_diem_tin_nhiem.csv", index=False, encoding="utf-8-sig")

temporary_figure, temporary_axis = plt.subplots()
credit.plot(kind="kde", ax=temporary_axis)
line = temporary_axis.lines[0]
x_data = line.get_xdata()
y_data = line.get_ydata()
plt.close(temporary_figure)

plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, color="#8e44ad", linewidth=3)
plt.fill_between(x_data, y_data, color="#8e44ad", alpha=0.22)
below_mask = x_data <= subprime_threshold
plt.fill_between(x_data[below_mask], y_data[below_mask], color="red", alpha=0.18, label="Vùng dưới ngưỡng (<600)")
plt.axvline(subprime_threshold, color="red", linestyle="--", linewidth=2.4, label=f"Ngưỡng tín dụng dưới chuẩn (<{subprime_threshold})")
plt.axvline(system_mean, color="blue", linestyle=":", linewidth=2.8, label=f"Trung bình hệ thống ({system_mean:.0f})")
plt.title(" MẬT ĐỘ PHÂN PHỐI ĐIỂM TÍN NHIỆM TRONG QUẦN THỂ", fontsize=16, fontweight="bold")
plt.xlabel("Điểm Tín Nhiệm (Credit Score)", fontsize=14)
plt.ylabel("Mật độ xác suất", fontsize=14)
plt.grid(color="#cccccc", linewidth=1, alpha=0.9)
plt.legend(loc="upper left", frameon=True, fontsize=12)
plt.text(0.72, 0.78, f"Số người dưới ngưỡng: {below_count}\nTỷ lệ: {below_rate:.2f}%",
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))
save_chart("06_kde_credit_score.png")

# 7. TƯƠNG QUAN THU NHẬP VỚI MỨC ĐỘ VUNG TAY CHI TIÊU
summary_7 = tracker.groupby("savings_goal_met", as_index=False).agg(
    count=("user_id", "count"),
    avg_income=("monthly_income", "mean"),
    avg_expense=("monthly_expense_total", "mean"),
    avg_savings_rate=("savings_rate", "mean"),
)
summary_7["status"] = summary_7["savings_goal_met"].map({0: "Không đạt mục tiêu tiết kiệm", 1: "Đạt mục tiêu tiết kiệm"})
summary_7.to_csv(out / "07_thu_nhap_va_bay_lam_phat_loi_song.csv", index=False, encoding="utf-8-sig")

scatter_df = tracker[["monthly_income", "monthly_expense_total", "savings_goal_met"]].replace([np.inf, -np.inf], np.nan).dropna()
x = scatter_df["monthly_income"].astype(float)
y = scatter_df["monthly_expense_total"].astype(float)
correlation = x.corr(y)
regression_coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = regression_coef[0] * line_x + regression_coef[1]

plt.figure(figsize=(9, 6))
for value, color, label in [
    (0, "#e75c5c", "Không đạt mục tiêu tiết kiệm"),
    (1, "#3ca9e0", "Đạt mục tiêu tiết kiệm"),
]:
    subset = scatter_df[scatter_df["savings_goal_met"] == value]
    plt.scatter(subset["monthly_income"], subset["monthly_expense_total"], alpha=0.55, color=color, label=label)
plt.title(" TƯƠNG QUAN THU NHẬP VỚI MỨC ĐỘ VUNG TAY CHI TIÊU", fontweight="bold")
plt.xlabel("Thu nhập hàng tháng")
plt.ylabel("Tổng chi tiêu hàng tháng")
plt.legend(loc="upper left")
save_chart("07_scatter_income_vs_expense.png")

# TÓM TẮT
summary_text = f"""TÓM TẮT KẾT QUẢ

1. CÁN CÂN THANH KHOẢN TIỀN MẶT NGẮN HẠN
- Tổng thu nhập: {total_income:,.2f}
- Tổng chi tiêu: {total_expenses:,.2f}
- Dòng tiền ròng: {net_cash_flow:,.2f}
- Tỷ lệ tiêu dùng: {consumption_ratio:.2f}%

2. BIÊN ĐỘ LỆCH RÒNG GIỮA THỰC CHI VỚI HẠN MỨC NGÂN SÁCH
- Xanh lá: chi dưới ngân sách.
- Đỏ: vượt ngân sách.

3. DANH MỤC LŨY KẾ TIÊU TỐN TIỀN MẶT DÀI HẠN
- Danh mục chi lớn nhất: {long_term.iloc[0]["Category"]}
- Tổng chi: {long_term.iloc[0]["Amount"]:,.2f}
- Tỷ trọng: {long_term.iloc[0]["share_percent"]:.2f}%

4. BIÊN ĐỘ CHI TIÊU THEO BIẾN ĐỊNH TÍNH ÁP LỰC VÀ NỢ
- Đỏ: Nguy hiểm, DTI > 35%.
- Xanh dương: An toàn, DTI <= 35%.
- Trục X: Low - High - Medium theo mẫu.

5. MA TRẬN TƯƠNG QUAN TUYẾN TÍNH ĐA BIẾN
- Heatmap đã có số trong từng ô.

6. MẬT ĐỘ PHÂN PHỐI ĐIỂM TÍN NHIỆM TRONG QUẦN THỂ
- Ngưỡng tín dụng dưới chuẩn: < {subprime_threshold}.
- Trung bình hệ thống: {system_mean:.2f}.
- Số người dưới ngưỡng: {below_count}.
- Tỷ lệ dưới ngưỡng: {below_rate:.2f}%.

7. TƯƠNG QUAN THU NHẬP VỚI MỨC ĐỘ VUNG TAY CHI TIÊU
- Đã thêm đường hồi quy.
- Hệ số tương quan r = {correlation:.2f}.
"""
(out / "tom_tat_ket_qua.txt").write_text(summary_text, encoding="utf-8")
print("Hoàn thành tạo lại toàn bộ biểu đồ và bảng kết quả.")