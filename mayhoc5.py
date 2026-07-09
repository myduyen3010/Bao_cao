import pandas as pd # Thư viện thao tác dữ liệu dạng bảng (Dataframe)
import numpy as np # Thư viện tính toán toán học và ma trận
import matplotlib.pyplot as plt # Thư viện vẽ đồ thị cơ bản
import seaborn as sns # Thư viện vẽ đồ thị thống kê đẹp mắt dựa trên matplotlib
import time # Thư viện đo thời gian thực thi
from pathlib import Path # Thư viện quản lý đường dẫn file/thư mục đa nền tảng

# Thư viện Scikit-Learn (Bộ công cụ Machine Learning)
from sklearn.model_selection import train_test_split # Hàm chia dữ liệu thành tập Train và Test
from sklearn.preprocessing import LabelEncoder, StandardScaler # Tiền xử lý: Mã hóa nhãn và Chuẩn hóa tỷ lệ dữ liệu
from sklearn.linear_model import LogisticRegression # Thuật toán Phân loại Hồi quy Logistic
from sklearn.tree import DecisionTreeClassifier # Thuật toán Phân loại Cây quyết định
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Thuật toán Rừng ngẫu nhiên
# Các hàm đo lường hiệu suất mô hình (Bổ sung classification_report và mean_squared_error)
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, mean_squared_error

# =========================================================================
# 0. KHỞI ĐỘNG HỆ THỐNG VÀ CÀI ĐẶT CẤU HÌNH
# =========================================================================
print("[*] Đang khởi động hệ thống phân tích...")
global_start_time = time.time() # Bắt đầu đo tổng thời gian chạy script

# Thiết lập thẩm mỹ cho đồ thị
plt.rcParams['font.family'] = 'DejaVu Sans' # Thiết lập font chữ mặc định cho đồ thị
sns.set_theme(style="whitegrid") # Đặt nền đồ thị có lưới màu trắng để dễ nhìn
out_dir = Path("python_finance_outputs") # Khai báo thư mục lưu kết quả đầu ra
out_dir.mkdir(exist_ok=True) # Tạo thư mục nếu chưa có

# =========================================================================
# 1. TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
# =========================================================================
try:
    df = pd.read_csv("Track_CLEANED.csv") # Đọc file dữ liệu csv
    print("[*] Đã nạp thành công dữ liệu gốc.")
except FileNotFoundError: # Xử lý ngoại lệ nếu không tìm thấy file để tránh chương trình bị crash
    print("Lỗi: Không tìm thấy file dữ liệu Track_CLEANED.csv!")
    exit() # Dừng chương trình nếu lỗi

# Lọc bỏ các dòng thiếu dữ liệu mục tiêu tiết kiệm
df = df.dropna(subset=['savings_goal_met', 'actual_savings'])

# 9 Biến đầu vào cơ sở (Base Features)
base_features = [
    "debt_to_income_ratio", "credit_score", "savings_rate", 
    "emergency_fund", "loan_payment", "monthly_income", 
    "monthly_expense_total", "discretionary_spending", "essential_spending"
]
# Lọc lại danh sách đặc trưng: Chỉ giữ lại những cột thực sự tồn tại trong Dataframe
base_features = [col for col in base_features if col in df.columns]

# Xử lý giá trị khuyết thiếu bằng Median
for col in base_features:
    if df[col].isnull().any():# Nếu cột có chứa giá trị NaN
        df[col] = df[col].fillna(df[col].median()) # Điền các ô trống bằng giá trị trung vị

X_base = df[base_features] # Ma trận chứa các biến đầu vào gốc

# Mục tiêu 1 cho Tầng Phân loại (0 = Không đạt, 1 = Đạt mục tiêu)
le = LabelEncoder() 
y_class = le.fit_transform(df['savings_goal_met']) 
target_names = le.classes_ # Lưu lại tên các nhãn gốc để dùng sau này

# Mục tiêu 2 cho Tầng Hồi quy (Dự đoán số tiền tiết kiệm thực tế)
y_reg = df['actual_savings']

# Chia tập Train/Test (Tỷ lệ 80/20, giữ phân tầng lớp)
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split( 
    X_base, y_class, y_reg, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_class 
)

# Chuẩn hóa dữ liệu gốc (Z-score Scaling)
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

# =========================================================================
# 2. TẦNG 1: HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC "CHUYÊN GIA" PHÂN LOẠI
# =========================================================================
print("\n[VÒNG 1] ĐÁNH GIÁ CÁC MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)")
print("-" * 70)

# Khai báo danh sách các mô hình học máy phân loại
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    "Balanced Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42)
}

train_probs_list = [] 
test_probs_list = []
hybrid_feature_names = base_features.copy() # Khởi tạo danh sách tên biến lai ghép

# Duyệt qua từng mô hình để huấn luyện, đo thời gian và in báo cáo chuẩn
for name, model in models.items():
    model_start = time.time() # Bắt đầu bấm giờ cho riêng mô hình này
    
    # Huấn luyện mô hình phân loại dựa trên tập Train
    model.fit(X_train_scaled, y_class_train)
    y_pred = model.predict(X_test_scaled)
    
    model_end = time.time() # Dừng bấm giờ mô hình
    
    # Xuất báo cáo y hệt định dạng mong muốn
    print(f"▶ {name} Report:")
    print(classification_report(y_class_test, y_pred, digits=2))
    print(f"Thời gian thực thi: {model_end - model_start:.2f} giây")
    print("-" * 70)

    # Trích xuất xác suất dự đoán (%) để làm đầu vào cho Tầng 2
    train_probs_list.append(model.predict_proba(X_train_scaled))
    test_probs_list.append(model.predict_proba(X_test_scaled))

    # Đặt tên cột mới cho các thuộc tính xác suất vừa sinh ra
    for class_name in target_names:
        hybrid_feature_names.append(f"Prob_{name.replace(' ', '_')}_{class_name}")

# =========================================================================
# 3. GIAI ĐOẠN XẾP CHỒNG ĐẶC TRƯNG (FEATURE STACKING)
# =========================================================================
print("\n[*] Đang gộp dữ liệu: Bơm các cột xác suất vào dữ liệu gốc...")
X_train_hybrid = np.hstack([X_train_scaled] + train_probs_list)
X_test_hybrid = np.hstack([X_test_scaled] + test_probs_list)

# =========================================================================
# 4. TẦNG 2: HUẤN LUYỆN "GIÁM ĐỐC" HỒI QUY (META-MODEL)
# =========================================================================
print("\n[VÒNG 2] DỰ ĐOÁN SỐ TIỀN TIẾT KIỆM THỰC TẾ (REGRESSION)")
print("-" * 70)

meta_start = time.time() # Bấm giờ cho tầng hồi quy

# Khởi tạo mô hình Random Forest dạng Hồi quy gồm 300 cây
regressor = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
regressor.fit(X_train_hybrid, y_reg_train) # Huấn luyện dựa trên tập dữ liệu lai ghép (gốc + xác suất)
reg_preds = regressor.predict(X_test_hybrid) # Dự đoán trên tập kiểm tra

# Đánh giá Meta-Model bằng các chỉ số cốt lõi
r2 = r2_score(y_reg_test, reg_preds) 
mae = mean_absolute_error(y_reg_test, reg_preds) 
rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds)) # Sai số bình phương trung bình (RMSE) giúp bắt lỗi lệch lớn

meta_end = time.time()

print(f"▶ Meta-Model RandomForest Regressor:")
print(f" -> Hệ số xác định (R² Score)        : {r2*100:.2f}%")
print(f" -> Sai số tiền dự đoán trung bình (MAE) : ${mae:.2f}")
print(f" -> Sai số căn lề bình phương (RMSE)  : ${rmse:.2f}")
print(f"Thời gian thực thi: {meta_end - meta_start:.2f} giây")
print("-" * 70)

# =========================================================================
# 5. XUẤT ĐỒ THỊ BÁO CÁO KẾT QUẢ VÀ TỔNG KẾT
# =========================================================================
print("\n[*] Đang khởi tạo và xuất đồ thị phân tích...")

# --- Đồ thị 1: Biểu đồ phân tán (Thực tế vs AI dự đoán) ---
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, reg_preds, alpha=0.6, color='#27ae60', edgecolor='w', s=60)
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
    if 'Prob_Logistic' in name: colors.append('#3498db') # Màu xanh biển cho Logistic
    elif 'Prob_Decision' in name: colors.append('#e67e22') # Màu cam cho Decision Tree
    elif 'Prob_Balanced' in name: colors.append('#e74c3c') # Màu đỏ cho Random Forest
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

# CHỐT TỔNG THỜI GIAN CHẠY CHƯƠNG TRÌNH
global_end_time = time.time()
print(f"\n[OK] Toàn bộ quá trình xử lý hoàn tất thành công!")
print(f" -> Thư mục lưu đồ thị báo cáo: {out_dir}")
print(f" -> Tổng thời gian thực thi toàn luồng dữ liệu: {global_end_time - global_start_time:.2f} giây.")