import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

out = Path("Financial_Visualization")
out.mkdir(exist_ok=True)

income = pd.read_csv("Income_CLEANED.csv")
expenses = pd.read_csv("Expenses_CLEANED.csv")
budget = pd.read_csv("Budget_CLEANED.csv")
tracker = pd.read_csv("Tracker_Clean.csv")
extended = pd.read_csv("8000_Extended_Clean")

for col in [
    "monthly_income", "monthly_expense_total", "credit_score",
    "debt_to_income_ratio", "discretionary_spending", "savings_goal_met"
]:
    if col in tracker.columns:
        tracker[col] = pd.to_numeric(tracker[col], errors="coerce")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


def save(name):
    plt.tight_layout()
    plt.savefig(out / name, dpi=300, bbox_inches="tight")
    plt.close()


def norm_cat(s):
    return str(s).strip().lower().replace("restuarant", "restaurant").replace("coffe", "cafe")


# 1. CÁN CÂN THANH KHOẢN TIỀN MẶT NGẮN HẠN
total_income = income["amount"].sum()
total_expenses = expenses["amount"].sum()
net_cash_flow = total_income - total_expenses
consumption_ratio = total_expenses / total_income * 100

pd.DataFrame({
    "metric": [
        "Tổng thu nhập",
        "Tổng chi tiêu",
        "Dòng tiền ròng",
        "Tỷ lệ tiêu dùng so với thu nhập (%)"
    ],
    "value": [total_income, total_expenses, net_cash_flow, consumption_ratio]
}).to_csv(out / "01_can_can_thanh_khoan.csv", index=False)

plt.figure(figsize=(9, 5.5))
labels = ["Tổng thu nhập", "Tổng chi tiêu", "Dòng tiền ròng"]
vals = [total_income, total_expenses, net_cash_flow]
colors = ["#2ca02c", "#d62728", "#1f77b4"]
bars = plt.bar(labels, vals, color=colors)
plt.title("1. CÁN CÂN THANH KHOẢN TIỀN MẶT NGẮN HẠN", fontweight="bold")
plt.ylabel("Số tiền")
for bar, value in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}", ha="center", va="bottom", fontweight="bold")
plt.text(0.5, max(vals) * 0.82, f"Tỷ lệ tiêu dùng: {consumption_ratio:.2f}%", ha="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
plt.legend(handles=[Patch(color=colors[0], label="Xanh lá: Thu nhập"), Patch(color=colors[1], label="Đỏ: Chi tiêu"), Patch(color=colors[2], label="Xanh dương: Dòng tiền ròng")], loc="upper right")
save("01_column_chart_can_can_thanh_khoan.png")

# 2. BIÊN ĐỘ LỆCH RÒNG GIỮA THỰC CHI VỚI HẠN MỨC NGÂN SÁCH
expenses["category_norm"] = expenses["category"].map(norm_cat)
budget["category_norm"] = budget["category"].map(norm_cat)
actual = expenses.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "actual_spending"})
planned = budget.groupby("category_norm", as_index=False)["amount"].sum().rename(columns={"amount": "budget_amount"})
budget_compare = planned.merge(actual, on="category_norm", how="outer").fillna(0)
budget_compare["variance"] = budget_compare["budget_amount"] - budget_compare["actual_spending"]
budget_compare["status"] = np.where(budget_compare["variance"] < 0, "Vượt ngân sách", "Chi dưới ngân sách")
budget_compare = budget_compare.sort_values("variance")
budget_compare.to_csv(out / "02_do_lech_ngan_sach.csv", index=False)
plt.figure(figsize=(10, max(7, len(budget_compare) * 0.33)))
plt.barh(budget_compare["category_norm"], budget_compare["variance"], color=np.where(budget_compare["variance"] >= 0, "#2ca02c", "#d62728"))
plt.axvline(0, color="black", linewidth=1)
plt.title("2. BIÊN ĐỘ LỆCH RÒNG GIỮA THỰC CHI VỚI HẠN MỨC NGÂN SÁCH", fontweight="bold")
plt.xlabel("Ngân sách - Thực chi")
plt.ylabel("Danh mục")
plt.legend(handles=[Patch(color="#2ca02c", label="Xanh lá: Chi dưới ngân sách"), Patch(color="#d62728", label="Đỏ: Vượt ngân sách")], loc="lower right")
save("02_diverging_bar_chart_ngan_sach.png")

# 3. DANH MỤC LŨY KẾ TIÊU TỐN TIỀN MẶT DÀI HẠN
ext_exp = extended.copy()
if "TransactionType" in ext_exp.columns:
    temp = ext_exp[ext_exp["TransactionType"].astype(str).str.lower().str.contains("expense|debit|spent|payment", na=False)]
    if not temp.empty:
        ext_exp = temp
long_term = ext_exp.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
long_term["share_percent"] = long_term["Amount"] / long_term["Amount"].sum() * 100
long_term.to_csv(out / "03_tich_luy_dai_han_theo_danh_muc.csv", index=False)
top = long_term.head(10)
gradient_colors = [plt.cm.Blues(0.9 - i * (0.45 / max(len(top) - 1, 1))) for i in range(len(top))]
plt.figure(figsize=(10, 6))
bars = plt.barh(top["Category"], top["Amount"], color=gradient_colors)
plt.gca().invert_yaxis()
plt.title("3. DANH MỤC LŨY KẾ TIÊU TỐN TIỀN MẶT DÀI HẠN", fontweight="bold")
plt.xlabel("Tổng chi tiêu tích lũy")
plt.ylabel("Danh mục")
for bar, value, percent in zip(bars, top["Amount"], top["share_percent"]):
    plt.text(value, bar.get_y() + bar.get_height() / 2, f" {value / 1e6:.2f}M ({percent:.1f}%)", va="center", fontsize=9)
save("03_bar_chart_tieu_dung_tich_luy.png")

# 4. BIÊN ĐỘ CHI TIÊU THEO BIẾN ĐỊNH TÍNH ÁP LỰC VÀ NỢ
stress_high = tracker["financial_stress_level"].astype(str).str.lower().str.contains("high|cao", na=False)
dti_threshold = tracker["debt_to_income_ratio"].quantile(0.75)
tracker["risk_group"] = np.where((tracker["debt_to_income_ratio"] >= dti_threshold) & stress_high, "DTI cao + stress cao", "Nhóm bình thường")
summary4 = tracker.groupby("risk_group", as_index=False).agg(
    count=("user_id", "count"),
    avg_dti=("debt_to_income_ratio", "mean"),
    low_q1=("discretionary_spending", lambda x: x.quantile(0.25)),
    avg_discretionary_spending=("discretionary_spending", "mean"),
    median_discretionary_spending=("discretionary_spending", "median"),
    high_q3=("discretionary_spending", lambda x: x.quantile(0.75)),
    avg_monthly_expense=("monthly_expense_total", "mean")
)
summary4.to_csv(out / "04_so_sanh_nhom_rui_ro.csv", index=False)
groups = ["Nhóm bình thường", "DTI cao + stress cao"]
box_data = [tracker.loc[tracker["risk_group"] == group, "discretionary_spending"].dropna() for group in groups]
plt.figure(figsize=(10, 6))
safe_limit = 400
danger_limit = 700
max_spending = tracker["discretionary_spending"].max()
plt.axhspan(0, safe_limit, color="#2ca02c", alpha=0.08)
plt.axhspan(danger_limit, max_spending, color="#d62728", alpha=0.08)
try:
    box = plt.boxplot(box_data, tick_labels=groups, patch_artist=True)
except TypeError:
    box = plt.boxplot(box_data, labels=groups, patch_artist=True)
for patch, color in zip(box["boxes"], ["#9ecae1", "#fdae6b"]):
    patch.set_facecolor(color)
plt.title("4. BIÊN ĐỘ CHI TIÊU THEO BIẾN ĐỊNH TÍNH ÁP LỰC VÀ NỢ", fontweight="bold")
plt.ylabel("Chi tiêu linh hoạt")
for i, data in enumerate(box_data, start=1):
    mean_value = data.mean()
    q1_value = data.quantile(0.25)
    q3_value = data.quantile(0.75)
    plt.scatter(i, mean_value, marker="D", s=60, color="black", zorder=3)
    plt.text(i, mean_value + 35, f"TB = {mean_value:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(i, q1_value - 35, f"Low = {q1_value:.2f}", ha="center", fontsize=9, color="green")
    plt.text(i, q3_value + 35, f"High = {q3_value:.2f}", ha="center", fontsize=9, color="red")
plt.axhline(safe_limit, color="#2ca02c", linestyle="--", linewidth=1)
plt.axhline(danger_limit, color="#d62728", linestyle="--", linewidth=1)
plt.text(2.18, safe_limit, "Vùng an toàn", color="green", fontsize=10, va="bottom")
plt.text(2.18, danger_limit, "Vùng nguy cơ cao", color="red", fontsize=10, va="bottom")
plt.text(0.02, 0.97, "CHÚ THÍCH:\nLow = Q1, nhóm 25% chi thấp\nTB = Trung bình\nHigh = Q3, nhóm 25% chi cao\nĐường giữa hộp = Trung vị\nChấm tròn = Giá trị ngoại lệ\nNền xanh = vùng an toàn\nNền đỏ = vùng nguy cơ cao", transform=plt.gca().transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9))
save("04_boxplot_no_stress.png")

# 5. MA TRẬN TƯƠNG QUAN TUYẾN TÍNH ĐA BIẾN
corr_cols = ["monthly_income", "monthly_expense_total", "savings_rate", "budget_goal", "credit_score", "debt_to_income_ratio", "loan_payment", "investment_amount", "emergency_fund", "transaction_count", "discretionary_spending", "essential_spending", "rent_or_mortgage", "actual_savings", "savings_goal_met"]
for col in corr_cols:
    tracker[col] = pd.to_numeric(tracker[col], errors="coerce")
corr = tracker[corr_cols].corr(numeric_only=True)
corr.to_csv(out / "05_ma_tran_tuong_quan.csv")
plt.figure(figsize=(12, 10))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, label="Hệ số tương quan")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("5. MA TRẬN TƯƠNG QUAN TUYẾN TÍNH ĐA BIẾN", fontweight="bold")
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        value = corr.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.55 else "black")
plt.text(0.01, -0.18, "Đỏ: tương quan dương | Xanh: tương quan âm | Trắng: gần 0", transform=plt.gca().transAxes)
save("05_heatmap_tuong_quan.png")

# 6. MẬT ĐỘ PHÂN PHỐI ĐIỂM TÍN NHIỆM & RỦI RO DƯỚI CHUẨN
credit = pd.to_numeric(tracker["credit_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
safe_threshold = 670
credit_mean = credit.mean()
below = int((credit < safe_threshold).sum())
below_rate = below / len(credit) * 100
pd.DataFrame({"metric": ["Điểm tín dụng trung bình", "Ngưỡng an toàn giả định", "Số người dưới ngưỡng", "Tỷ lệ dưới ngưỡng (%)"], "value": [credit_mean, safe_threshold, below, below_rate]}).to_csv(out / "06_phan_phoi_diem_tin_nhiem.csv", index=False)
plt.figure(figsize=(9, 5.5))
credit.plot(kind="kde", label="Đường mật độ KDE", linewidth=2)
ymax = plt.gca().get_ylim()[1]
plt.axvline(credit_mean, linestyle="-.", linewidth=2, label=f"Điểm tín dụng TB: {credit_mean:.2f}")
plt.axvline(safe_threshold, linestyle="--", linewidth=2, label=f"Ngưỡng an toàn giả định: {safe_threshold}")
plt.fill_betweenx([0, ymax], credit.min(), safe_threshold, alpha=0.12, label="Vùng dưới ngưỡng")
plt.title("6. MẬT ĐỘ PHÂN PHỐI ĐIỂM TÍN NHIỆM & RỦI RO DƯỚI CHUẨN", fontweight="bold")
plt.xlabel("Credit score")
plt.ylabel("Mật độ")
plt.text(0.03, 0.95, f"Điểm tín dụng TB: {credit_mean:.2f}\nNgưỡng an toàn: {safe_threshold}\nSố người dưới ngưỡng: {below}\nTỷ lệ dưới ngưỡng: {below_rate:.2f}%", transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
plt.legend(loc="upper left")
save("06_kde_credit_score.png")

# 7. TƯƠNG QUAN THU NHẬP VỚI MỨC ĐỘ VUNG TAY CHI TIÊU
summary7 = tracker.groupby("savings_goal_met", as_index=False).agg(count=("user_id", "count"), avg_income=("monthly_income", "mean"), avg_expense=("monthly_expense_total", "mean"), avg_savings_rate=("savings_rate", "mean"))
summary7["status"] = summary7["savings_goal_met"].map({0: "Không đạt mục tiêu tiết kiệm", 1: "Đạt mục tiêu tiết kiệm"})
summary7.to_csv(out / "07_thu_nhap_va_bay_lam_phat_loi_song.csv", index=False)
scatter_df = tracker[["monthly_income", "monthly_expense_total", "savings_goal_met"]].replace([np.inf, -np.inf], np.nan).dropna()
x = scatter_df["monthly_income"].astype(float)
y = scatter_df["monthly_expense_total"].astype(float)
r = x.corr(y)
coef = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = coef[0] * line_x + coef[1]
plt.figure(figsize=(9, 6))
for value, color, label in [(0, "#1f77b4", "Không đạt mục tiêu tiết kiệm"), (1, "#ff7f0e", "Đạt mục tiêu tiết kiệm")]:
    subset = scatter_df[scatter_df["savings_goal_met"] == value]
    plt.scatter(subset["monthly_income"], subset["monthly_expense_total"], alpha=0.55, color=color, label=label)
plt.plot(line_x, line_y, color="black", linewidth=2, label=f"Đường hồi quy | r = {r:.2f}")
plt.title("7. TƯƠNG QUAN THU NHẬP VỚI MỨC ĐỘ VUNG TAY CHI TIÊU", fontweight="bold")
plt.xlabel("Thu nhập hàng tháng")
plt.ylabel("Tổng chi tiêu hàng tháng")
plt.legend(loc="upper left")
plt.text(0.02, 0.05, "Diễn giải: thu nhập tăng kèm chi tiêu tăng\ncó thể cho thấy xu hướng vung tay chi tiêu.", transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
save("07_scatter_income_vs_expense.png")

print("Hoàn thành tạo lại biểu đồ.")
