import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Thư viện Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, classification_report

# Thiết lập thẩm mỹ cho đồ thị
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid")
out_dir = Path("python_finance_outputs")
out_dir.mkdir(exist_ok=True)

print("=== HỆ THỐNG AI DỰ ĐOÁN MỤC TIÊU TIẾT KIỆM (STACKING MODEL) ===\n")

try:
    df = pd.read_csv("Track_CLEANED.csv")
    print("[*] Đã nạp thành công dữ liệu gốc.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu Track_CLEANED.csv!")
    exit()

# Lọc bỏ các dòng thiếu dữ liệu mục tiêu tiết kiệm
df = df.dropna(subset=['savings_goal_met', 'actual_savings'])

# 9 Biến đầu vào cơ sở (Base Features)
base_features = [
    "debt_to_income_ratio", "credit_score", "savings_rate", 
    "emergency_fund", "loan_payment", "monthly_income", 
    "monthly_expense_total", "discretionary_spending", "essential_spending"
]
base_features = [col for col in base_features if col in df.columns]

# Xử lý giá trị khuyết thiếu bằng Median
for col in base_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

X_base = df[base_features]

# --- SỰ THAY ĐỔI CHÍNH NẰM Ở ĐÂY ---
# Mục tiêu 1 cho Tầng Phân loại (0 = Không đạt, 1 = Đạt mục tiêu)
y_class = df['savings_goal_met'].astype(int)
target_names = ['Khong_Dat', 'Dat_Muc_Tieu']

# Mục tiêu 2 cho Tầng Hồi quy (Dự đoán số tiền tiết kiệm thực tế)
y_reg = df['actual_savings']

# Chia tập Train/Test
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X_base, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# Chuẩn hóa dữ liệu gốc
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================================
# 2. TẦNG 1: HUẤN LUYỆN 3 "CHUYÊN GIA" PHÂN LOẠI
# =========================================================================
print("\n[VÒNG 1] Dự đoán khả năng ĐẠT MỤC TIÊU TIẾT KIỆM (0/1)...")

models = {
    "Logistic": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    "Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42)
}

train_probs_list = []
test_probs_list = []
hybrid_feature_names = base_features.copy()

for name, model in models.items():
    model.fit(X_train_scaled, y_class_train)
    acc = accuracy_score(y_class_test, model.predict(X_test_scaled))
    print(f"  -> Độ chính xác của chuyên gia {name}: {acc*100:.2f}%")
    
    train_probs_list.append(model.predict_proba(X_train_scaled))
    test_probs_list.append(model.predict_proba(X_test_scaled))
    
    for class_name in target_names:
        hybrid_feature_names.append(f"Prob_{name}_{class_name}")


# =========================================================================
# 3. GIAI ĐOẠN XẾP CHỒNG (STACKING / FEATURE ENGINEERING)
# =========================================================================
print("\n[*] Đang gộp dữ liệu: Bơm các cột xác suất vào dữ liệu gốc...")
X_train_hybrid = np.hstack([X_train_scaled] + train_probs_list)
X_test_hybrid = np.hstack([X_test_scaled] + test_probs_list)


# =========================================================================
# 4. TẦNG 2: HUẤN LUYỆN "GIÁM ĐỐC" HỒI QUY (META-MODEL)
# =========================================================================
print("\n[VÒNG 2] Dự đoán SỐ TIỀN TIẾT KIỆM THỰC TẾ (Actual Savings)...")
regressor = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
regressor.fit(X_train_hybrid, y_reg_train)

reg_preds = regressor.predict(X_test_hybrid)
r2 = r2_score(y_reg_test, reg_preds)
mae = mean_absolute_error(y_reg_test, reg_preds)

print(f"\n============================================================")
print(f"KẾT QUẢ DỰ ĐOÁN SỐ TIỀN TIẾT KIỆM (ACTUAL SAVINGS):")
print(f" -> Hệ số xác định (R² Score): {r2*100:.2f}%")
print(f" -> Sai số tiền dự đoán trung bình (MAE): ${mae:.2f}")
print(f"============================================================\n")


# =========================================================================
# 5. XUẤT ĐỒ THỊ BÁO CÁO KẾT QUẢ
# =========================================================================

# --- Đồ thị 1: Biểu đồ phân tán (Thực tế vs AI dự đoán) ---
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, reg_preds, alpha=0.6, color='#27ae60', edgecolor='w', s=60) # Đổi sang màu xanh lá (tiền bạc)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2.5)
plt.title('HÌNH 1: SỐ TIỀN TIẾT KIỆM THỰC TẾ vs AI DỰ ĐOÁN', fontweight='bold', pad=15)
plt.xlabel('Tiết kiệm thực tế (Actual Savings)', fontweight='bold')
plt.ylabel('Hệ thống AI dự đoán (Predicted Savings)', fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'savings_scatter_plot.png', dpi=300)
plt.close()

# --- Đồ thị 2: Đánh giá tầm quan trọng của các Biến ---
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
colors = []
for i in indices:
    name = hybrid_feature_names[i]
    if 'Prob_Logistic' in name: colors.append('#3498db')
    elif 'Prob_Tree' in name: colors.append('#e67e22')
    elif 'Prob_Forest' in name: colors.append('#e74c3c')
    else: colors.append('#27ae60') # Màu xanh lá cho dữ liệu gốc

plt.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black', height=0.6)
plt.yticks(range(len(indices)), [hybrid_feature_names[i] for i in indices], fontsize=9)
plt.title('HÌNH 2: TẦM QUAN TRỌNG CỦA CÁC YẾU TỐ ĐỐI VỚI TIẾT KIỆM', fontweight='bold', pad=15)
plt.xlabel('Tỷ lệ đóng góp vào quyết định cuối cùng (%)', fontweight='bold')

for index, value in enumerate(importances[indices]):
    plt.text(value + 0.001, index, f"{value*100:.1f}%", va='center', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / 'savings_feature_importance.png', dpi=300)
plt.close()

print(f"[OK] Hoàn tất! Báo cáo hình ảnh đã được lưu tại thư mục: {out_dir}")