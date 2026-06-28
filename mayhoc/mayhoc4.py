# ==============================================================================
# MÃ NGUỒN PIPELINE AI CHUYÊN SÂU NÂNG CẤP - THÀNH VIÊN 05
# TỐI ƯU HÓA KHẢ NĂNG DỰ ĐOÁN CHO DỮ LIỆU MẤT CÂN BẰNG (SMOTE & CLASS WEIGHT)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Thư viện xử lý và huấn luyện AI chuyên dụng (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Thư viện tối ưu cân bằng dữ liệu
from imblearn.over_sampling import SMOTE

# Thiết lập thẩm mỹ cho đồ thị báo cáo
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid")

print("==============================================================================")
print("🤖 [NGƯỜI 5]: KHỞI CHẠY PIPELINE AI TOÀN DIỆN (NÂNG CẤP) - DỰ BÁO ÁP LỰC TÀI CHÍNH")
print("==============================================================================")

# Tạo thư mục đầu ra riêng cho Chương 3
out_dir = Path("python_finance_outputs")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# 1. NẠP DỮ LIỆU SẠCH 
# ------------------------------------------------------------------------------
try:
    df = pd.read_csv("baocaoNNLT/python_finance_outputs/Track_CLEANED.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("Track_CLEANED.csv")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file dữ liệu sạch Track_CLEANED.csv!")
        exit()

# ------------------------------------------------------------------------------
# 2. TIỀN XỬ LÝ DỮ LIỆU & CÂN BẰNG MẪU (Data Preprocessing & Balancing)
# ------------------------------------------------------------------------------
print("\n[Bước 1] Tiến hành tiền xử lý, điền khuyết và mã hóa dữ liệu...")

# Loại bỏ dòng trống ở cột mục tiêu trước
df = df.dropna(subset=['financial_stress_level'])

# 2.1 Mã hóa biến mục tiêu định tính (Low, Medium, High) -> (0, 1, 2)
le = LabelEncoder()
df['target'] = le.fit_transform(df['financial_stress_level'])
target_names = list(le.classes_)

# 2.2 Lựa chọn đặc trưng
features = [
    'monthly_income', 'monthly_expense_total', 'savings_rate', 
    'credit_score', 'debt_to_income_ratio', 'discretionary_spending'
]
features = [col for col in features if col in df.columns]

# Điền giá trị trung bình cho các ô trống trong các cột đặc trưng (Thay vì dropna toàn bộ)
for col in features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

X = df[features]
y = df['target']

# Kiểm tra phân phối các lớp trước khi xử lý
print("📊 Phân bổ dữ liệu gốc ban đầu:")
for name, code in zip(target_names, le.transform(target_names)):
    print(f"  - Nhóm {name}: {sum(y == code)} mẫu")

# 2.3 Chia tập dữ liệu Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🚀 KỸ THUẬT NÂNG CAO: Áp dụng SMOTE để tái cân bằng tập huấn luyện (Train)
print("⚖️ Đang thực hiện thuật toán SMOTE để cân bằng dữ liệu học...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("📊 Phân bổ dữ liệu sau khi áp dụng SMOTE (Tập Train):")
for name, code in zip(target_names, le.transform(target_names)):
    print(f"  - Nhóm {name}: {sum(y_train_resampled == code)} mẫu")

# 2.4 Chuẩn hóa dữ liệu dựa trên tập đã cân bằng mẫu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# 3. HUÂN LUYỆN ĐỒNG THỜI VÀ SO SÁNH 3 MÔ HÌNH THUẬT TOÁN (ĐÃ THÊM CLASS_WEIGHT)
# ------------------------------------------------------------------------------
print("\n[Bước 2] Đang kích hoạt huấn luyện 3 thuật toán học máy nâng cao...")

models = {
    # Thêm tham số class_weight='balanced' để tối ưu hóa việc phân tách ranh giới các nhóm lệch
    "Logistic Regression (Tuyến tính)": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "Decision Tree (Cây quyết định)": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    "Random Forest (Rừng ngẫu nhiên)": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
}

best_model_name = ""
best_macro_f1 = 0  # Chuyển sang đánh giá bằng Macro F1-score vì dữ liệu thực tế bị lệch nhãn
best_model_obj = None
best_accuracy = 0
results_text = "=== BÁO CÁO KẾT QUẢ HUẤN LUYỆN AI CHUYÊN SÂU (CHƯƠNG 3) ===\n\n"

for name, model in models.items():
    # Huấn luyện trên tập dữ liệu đã qua xử lý SMOTE
    model.fit(X_train_scaled, y_train_resampled)
    
    # Dự đoán trên tập kiểm tra thực tế (giữ nguyên không SMOTE tập test)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    macro_f1 = report_dict['macro avg']['f1-score'] # Chỉ số vàng đánh giá dữ liệu mất cân bằng
    
    results_text += f"■ Mô hình: {name}\n"
    results_text += f"  - Độ chính xác tổng thể (Accuracy): {acc*100:.2f}%\n"
    results_text += f"  - Khả năng nhận diện cân bằng (Macro F1-Score): {macro_f1*100:.2f}%\n"
    results_text += f"  - Chi tiết chỉ số:\n{classification_report(y_test, y_pred, target_names=target_names)}\n"
    results_text += "-"*50 + "\n"
    
    # Lựa chọn mô hình tốt nhất dựa trên khả năng phân loại đều các lớp (Macro F1)
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_accuracy = acc
        best_model_name = name
        best_model_obj = model

with open(out_dir / "03_bao_cao_model_ai.txt", "w", encoding="utf-8") as f:
    f.write(results_text)

print(f"🎯 THUẬT TOÁN CHIẾN THẮNG THỰC SỰ: {best_model_name} với Accuracy = {best_accuracy*100:.2f}% (Macro F1 = {best_macro_f1*100:.2f}%)")

# ------------------------------------------------------------------------------
# 4. TRỰC QUAN HÓA KẾT QUẢ ĐÁNH GIÁ (Xuất đồ thị cho Word)
# ------------------------------------------------------------------------------
print("\n[Bước 3] Đang xuất đồ thị phân tích chuyên sâu...")

# Đồ thị 1: Ma trận nhầm lẫn (Confusion Matrix) của mô hình tốt nhất
best_y_pred = best_model_obj.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(6.5, 5))
# Sử dụng bảng màu GnBu/Oranges trực quan hơn
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=target_names, yticklabels=target_names,
            cbar=False, annot_kws={"weight": "bold", "size": 11})
plt.title(f'HÌNH 3.1: MA TRẬN NHẦM LẪN CỦA MÔ HÌNH {best_model_name.upper()}', fontweight='bold', fontsize=10, pad=12)
plt.xlabel('Hệ thống AI dự đoán (Predicted)', fontweight='bold', fontsize=9)
plt.ylabel('Thực tế hồ sơ (Actual)', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / 'hinh_3_1_confusion_matrix.png', dpi=300)
plt.close()

# Đồ thị 2: Đánh giá tầm quan trọng của các biến số (Feature Importance)
if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(8, 4.5))
    plt.barh(range(len(indices)), importances[indices], color='#2c3e50', edgecolor='black', height=0.5)
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=9)
    plt.title('HÌNH 3.2: ĐỘ QUAN TRỌNG CỦA CÁC BIẾN SỐ ĐẦU VÀO ĐỐI VỚI ÁP LỰC TÀI CHÍNH', fontweight='bold', fontsize=10, pad=12)
    plt.xlabel('Tỷ lệ đóng góp vào quyết định của AI (Weight)', fontweight='bold', fontsize=9)
    
    for index, value in enumerate(importances[indices]):
        plt.text(value + 0.01, index, f"{value*100:.1f}%", va='center', fontweight='bold', fontsize=8)
        
    plt.xlim(0, max(importances) * 1.18)
    plt.tight_layout()
    plt.savefig(out_dir / 'hinh_3_2_feature_importance.png', dpi=300)
    plt.close()

print(f"💾 SYSTEM: Đã cập nhật toàn bộ tài nguyên cải tiến vào thư mục: '{out_dir}/'")
print("==============================================================================")