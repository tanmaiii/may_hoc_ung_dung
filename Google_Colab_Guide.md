# 🚀 HƯỚNG DẪN CHẠY PROJECT TRÊN GOOGLE COLAB

## 📋 Tổng quan
Hướng dẫn chi tiết để chạy project **Telco Customer Churn Prediction** trên Google Colab một cách đơn giản và hiệu quả.

## ⚡ Ưu điểm của Google Colab
- ✅ **Miễn phí** và có GPU/TPU
- ✅ **Không cần cài đặt** gì trên máy tính
- ✅ **Đã có sẵn** tất cả thư viện cần thiết
- ✅ **Dễ chia sẻ** và collaborate
- ✅ **Lưu trữ trên Google Drive**

## 🔧 Bước 1: Chuẩn bị

### 1.1 Tạo Google Colab Notebook
1. Truy cập [Google Colab](https://colab.research.google.com/)
2. Đăng nhập tài khoản Google
3. Tạo notebook mới: **File > New notebook**

### 1.2 Upload dữ liệu
```python
import pandas as pd
# Nếu có liên kết tải xuống trực tiếp
url = "https://raw.githubusercontent.com/tanmaiii/may_hoc_ung_dung/refs/heads/main/telco-customer-churn.csv"
df = pd.read_csv(url)
```

## 📊 Bước 2: Notebook hoàn chỉnh

### 2.1 Setup và Import Libraries
```python
# Cài đặt thêm gói thư viện nếu cần
!pip install plotly seaborn

# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Máy học
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Tắt cảnh báo
import warnings
warnings.filterwarnings('ignore')

# Cài đặt style cho biểu đồ
plt.style.use('default')
sns.set_palette("husl")
```

### 2.2 Load và Explore Data (Load dữ liệu)
```python
# Tải dữ liệu
df = pd.read_csv('telco-customer-churn.csv')  # Hoặc đường dẫn file bạn upload

print("🎯 TỔNG QUAN DỮ LIỆU")
print(f"Kích thước: {df.shape}")
print(f"Giá trị thiếu: {df.isnull().sum().sum()}")
print("\n📊 5 dòng đầu tiên:")
df.head()
```

### 2.3 Data Analysis và Visualization (Phân tích dữ liệu và trực quan hóa)
```python
# Thông tin cơ bản
print("📈 THÔNG TIN DỮ LIỆU")
df.info()

print("\n📊 THỐNG KÊ MÔ TẢ")
df.describe()

# Phân bố Churn
print("\n🎯 PHÂN BỐ CHURN")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Tỷ lệ khách hàng rời bỏ dịch vụ (Churn rate): {churn_counts[1] / len(df) * 100:.2f}%")

# Trực quan hóa
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Biểu đồ tròn
axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%')
axes[0,0].set_title('Tỷ lệ khách hàng rời bỏ (Churn)')

# Biểu đồ phân phối
axes[0,1].hist(df['tenure'], bins=30, alpha=0.7)
axes[0,1].set_title('Phân bố thời gian sử dụng')
axes[0,1].set_xlabel('Số tháng sử dụng (tenure)')

# Biểu đồ hộp
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0])
axes[1,0].set_title('Chi phí hàng tháng theo Churn')

# Biểu đồ cột
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Loại hợp đồng và Churn')
axes[1,1].legend(title='Churn')

plt.tight_layout()
plt.show()
```

## 🔄 Bước 3: Data Preprocessing (Tiền xử lý dữ liệu)

### 3.1 Data Cleaning (Làm sạch)
```python
# Tạo bản sao để xử lý
df_clean = df.copy()

print("🧹 LÀM SẠCH DỮ LIỆU...")

# Xử lý TotalCharges (có giá trị ' ')
print(f"Giá trị ' ' trong TotalCharges: {(df_clean['TotalCharges'] == ' ').sum()}")
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Điền giá trị thiếu
df_clean['TotalCharges'].fillna(0, inplace=True)

# Loại bỏ customerID
df_clean = df_clean.drop('customerID', axis=1)

print(f"✅ Kích thước sau làm sạch: {df_clean.shape}")
print(f"Giá trị thiếu sau làm sạch: {df_clean.isnull().sum().sum()}")
```

### 3.2 Feature Engineering 
```python
print("🔧 TẠO ĐẶC TRƯNG MỚI...")

# Kiểm tra các giá trị unique trong tất cả các cột object
print("📊 KIỂM TRA CÁC GIÁ TRỊ UNIQUE TRONG CÁC CỘT:")
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        unique_vals = df_clean[col].unique()
        print(f"{col}: {unique_vals}")

# Xử lý các cột có giá trị đặc biệt (như "No phone service", "No internet service")
special_service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in special_service_cols:
    if col in df_clean.columns:
        print(f"\n🔧 Xử lý cột {col}...")
        # Chuyển "No phone service" và "No internet service" thành "No"
        df_clean[col] = df_clean[col].replace({'No phone service': 'No', 
                                             'No internet service': 'No'})
        print(f"Giá trị sau xử lý: {df_clean[col].unique()}")

# Mã hóa nhị phân cho các cột Yes/No
binary_cols = []
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' and col != 'Churn':
        unique_vals = df_clean[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            binary_cols.append(col)
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

print(f"\n✅ Các cột đã mã hóa nhị phân: {binary_cols}")

# Mã hóa one-hot cho các biến định danh
nominal_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
df_encoded = pd.get_dummies(df_clean, columns=nominal_cols, drop_first=True)

# Trả về 1, 0 thay vì yes, no
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Tạo các đặc trưng mới
df_encoded['AvgChargePerTenure'] = np.where(
    df_encoded['tenure'] > 0,
    df_encoded['TotalCharges'] / df_encoded['tenure'],
    df_encoded['MonthlyCharges']
)

# Nhóm thời gian sử dụng - tạo nhóm theo thời gian sử dụng 
# 0: 0-12m (Khách hàng mới)
# 1: 13-36m (Khách hàng trung hạn)
# 2: 37-60m (Khách hàng dài hạn)
# 3: 60m+ (Khách hàng trung thành)

try:
    df_encoded['TenureGroup'] = pd.cut(
        df_encoded['tenure'],
        bins=[0, 12, 36, 60, float('inf')],
        labels=['0-12m', '13-36m', '37-60m', '60m+']
    )
    # Mã hóa nhãn phân loại thành số
    df_encoded['TenureGroup'] = df_encoded['TenureGroup'].cat.codes
except Exception as e:
    print(f"Lỗi khi tạo TenureGroup: {e}")
    # Phương án dự phòng: tạo nhóm đơn giản
    df_encoded['TenureGroup'] = pd.cut(
        df_encoded['tenure'],
        bins=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

# Kiểm tra các cột còn lại chưa được mã hóa
print(f"\n📋 KIỂM TRA CÁC CỘT SAU KHI MÃ HÓA:")
remaining_object_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
if remaining_object_cols:
    print(f"⚠️  Các cột chưa được mã hóa: {remaining_object_cols}")
    for col in remaining_object_cols:
        print(f"   {col}: {df_encoded[col].unique()}")
else:
    print("✅ Tất cả các cột đã được mã hóa thành số!")

print(f"\n✅ Kích thước cuối cùng: {df_encoded.shape}")
df_encoded.head()
```

## 🎯 Bước 4: Feature Selection

### 4.1 Correlation Analysis
```python
# Bản đồ nhiệt tương quan
plt.figure(figsize=(20, 16))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Ma trận tương quan đặc trưng')
plt.show()

# Tương quan cao nhất với mục tiêu
target_corr = correlation_matrix['Churn'].abs().sort_values(ascending=False)
print("🎯 TOP 15 ĐẶC TRƯNG TƯƠNG QUAN VỚI CHURN:")
print(target_corr.head(15))
```

### 4.2 Feature Selection Methods
```python
# Chuẩn bị dữ liệu
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Kích thước đặc trưng: {X.shape}")
print(f"Phân bố target: {y.value_counts().to_dict()}")

# Phương pháp 1: SelectKBest
def select_features_univariate(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': scores
    }).sort_values('score', ascending=False)
    
    return selected_features, feature_scores

# Phương pháp 2: Random Forest Feature Importance
def select_features_rf(X, y, k=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance.head(k)['feature'].tolist()
    return selected_features, feature_importance

# Áp dụng lựa chọn đặc trưng
for k in [5, 10, 15]:
    print(f"\n🔍 TOP {k} ĐẶC TRƯNG:")
    
    # Lựa chọn đơn biến
    features_uni, scores_uni = select_features_univariate(X, y, k)
    print(f"Phương pháp Univariate: {features_uni}")
    
    # Random Forest
    features_rf, scores_rf = select_features_rf(X, y, k)
    print(f"Random Forest: {features_rf}")
```

## 🤖 Bước 5: Model Training và Evaluation

### 5.1 Data Splitting và Scaling
```python
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Tập huấn luyện: {X_train_scaled.shape}")
print(f"Tập kiểm tra: {X_test_scaled.shape}")
print(f"Phân bố target trong tập train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
```

### 5.2 Model Training
```python
# Import các thư viện cần thiết cho thuật toán
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

class MutilModel:
    def __init__(self):
        self.models = {
            # Hồi quy tuyến tính
            'LogisticRegression': LogisticRegression(
                solver='liblinear',   # tốt cho dữ liệu nhỏ, phân loại nhị phân
                C=1.0,                # độ phạt (regularization) - nhỏ hơn → chống overfit
                random_state=42
            ),
            # Rừng ngẫu nhiên
            'RandomForest': RandomForestClassifier(
                n_estimators=200,     # số lượng cây quyết định
                max_depth=10,         # giới hạn độ sâu cây để tránh overfitting
                min_samples_split=5,  # số mẫu ít nhất của 1 nút
                random_state=42
            ),
            # Máy vector hổ trợ (SVC là biến thể của SVM)
            'SVM': SVC(
                kernel='rbf',          # kernel phổ biến nhất
                C=1.0,                 # penalty, điều chỉnh biên độ margin
                gamma='scale',         # tự động điều chỉnh theo số chiều
                probability=True,      # Tính xác xuất dự đoán
                random_state=42
            ),
            # Láng giềng gần nhất
            'KNN': KNeighborsClassifier(
                n_neighbors=7,         # chọn số lân cận cầdn xét
                weights='distance',    # lân cận gần hơn có trọng số lớn hơn
                metric='minkowski'     # Khoảng cách Minkowski
            ),
            # Mạng nơron
           'NeuralNetwork': MLPClassifier(
              hidden_layer_sizes=(128, 64, 32),  # tăng số tầng ẩn, giảm dần số neuron
              activation='relu',                # relu vẫn là tốt nhất với dữ liệu phi tuyến
              solver='adam',                    # ổn định và nhanh
              alpha=0.0005,                     # hệ số regularization (L2), chống overfitting
              learning_rate='adaptive',        # giảm learning rate khi gặp khó
              learning_rate_init=0.001,        # learning rate khởi tạo
              early_stopping=True,             # dừng sớm nếu không cải thiện
              validation_fraction=0.1,         # 10% dữ liệu để validation khi training
              max_iter=500,                    # số vòng lặp (thường không cần quá lớn nếu early stopping)
              random_state=42
          )
        }

    def train(self, X_train, y_train):
        for model in self.models.values():
            model.fit(X_train, y_train)

    def evaluation(self, X_test, y_test, X_train=None, y_train=None, cv=5):
        report = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            if X_train is not None and y_train is not None:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean = None
                cv_std = None

            report[name] = {
                'accuracy': accuracy_score(y_test, y_pred), # Độ chính xác
                'precision': precision_score(y_test, y_pred), # Độ chính xác theo dự đoán
                'recall': recall_score(y_test, y_pred), # Khả năng phát hiện đúng
                'f1_score': f1_score(y_test, y_pred), # Chỉ số cân bằng giữa Precision và Recall
                'auc': auc, # khả năng mô hình phân biệt
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

        return report

    def predict_customer(self, model_name, customer_data, reference_columns, scaler):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        # Chuẩn bị dữ liệu đầu vào
        df_input = prepare_input(customer_data, reference_columns)
        df_scaled = scaler.transform(df_input)

        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

        return {
            "prediction": prediction,
            "probability": probability
        }

    def predict_all_customers(self, customer_data, reference_columns, scaler):
        # Chuẩn bị DataFrame đầu vào chung
        df_input = prepare_input(customer_data, reference_columns)
        df_scaled = scaler.transform(df_input)

        print("------------------------------🚀🚀🚀-----------------------------")

        # Lặp qua tất cả mô hình
        for name, model in self.models.items():
            pred = model.predict(df_scaled)[0]
            proba = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

            # In kết quả
            print(f"\n🔍 Kết quả của mô hình {name}:")
            print(f"-> {'🛡️ Ở lại' if pred == 0 else '🚶‍➡️ Rời đi'}")
            if proba is not None:
                print(f"  - Xác suất churn: {proba:.4f}")
            else:
                print("  - Xác suất churn: Không có (model không hỗ trợ predict_proba)")

    def predict_customer(self, model_name, customer_data, reference_columns, scaler):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        # Chuẩn bị dữ liệu đầu vào
        df_input = prepare_input(customer_data, reference_columns)
        df_scaled = scaler.transform(df_input)

        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

        return {
            "prediction": prediction,
            "probability": probability
        }

    def predict_all_customers(self, customer_data, reference_columns, scaler):
        """
        Dự đoán cho cùng 1 bộ dữ liệu đầu vào (customer_data) với tất cả các model đã lưu.
        In ra kết quả prediction và probability (nếu có) của từng model.
        """
        # Chuẩn bị DataFrame đầu vào chung
        df_input = prepare_input(customer_data, reference_columns)
        df_scaled = scaler.transform(df_input)

        print("------------------------------🚀🚀🚀-----------------------------")

        # Lặp qua tất cả mô hình
        for name, model in self.models.items():
            pred = model.predict(df_scaled)[0]
            proba = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

            # In kết quả
            print(f"\n🔍 Kết quả của mô hình {name}:")
            print(f"-> {'🛡️ Ở lại' if pred == 0 else '🚶‍➡️ Rời đi'}")
            if proba is not None:
                print(f"  - Xác suất churn: {proba:.4f}")
            else:
                print("  - Xác suất churn: Không có (model không hỗ trợ predict_proba)")
```

#### Chạy model

```python
# Khởi tạo và sử dụng MutilModel
print("🤖 KHỞI TẠO VÀ HUẤN LUYỆN CÁC MÔ HÌNH:")
multi_model = MutilModel()

print("📋 Các mô hình được sử dụng:")
for name in multi_model.models.keys():
    print(f"   - {name}")

# Huấn luyện tất cả các mô hình
print(f"\n🔄 Đang huấn luyện {len(multi_model.models)} mô hình...")
multi_model.train(X_train_scaled, y_train)
print("✅ Hoàn thành huấn luyện!")

# Đánh giá các mô hình
print(f"\n📊 ĐÁNH GIÁ HIỆU SUẤT:")
results = multi_model.evaluation(X_test_scaled, y_test, X_train_scaled, y_train)

# In kết quả chi tiết
for name, metrics in results.items():
    print(f"\n🔹 {name}:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}") 
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"   AUC:       {metrics['auc']:.4f}")
    if metrics['cv_mean'] is not None:
        print(f"   CV Score:  {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
```

```
🤖 KHỞI TẠO VÀ HUẤN LUYỆN CÁC MÔ HÌNH:
📋 Các mô hình được sử dụng:
   - LogisticRegression
   - RandomForest
   - SVM
   - KNN
   - NeuralNetwork

🔄 Đang huấn luyện 5 mô hình...
✅ Hoàn thành huấn luyện!

📊 ĐÁNH GIÁ HIỆU SUẤT:

🔹 LogisticRegression:
   Accuracy:  0.8197
   Precision: 0.6820
   Recall:    0.5979
   F1-Score:  0.6371
   AUC:       0.8626
   CV Score:  0.8005 (±0.0078)

🔹 RandomForest:
   Accuracy:  0.8148
   Precision: 0.7074
   Recall:    0.5121
   F1-Score:  0.5941
   AUC:       0.8615
   CV Score:  0.7984 (±0.0108)

🔹 SVM:
   Accuracy:  0.8112
   Precision: 0.6864
   Recall:    0.5282
   F1-Score:  0.5970
   AUC:       0.8214
   CV Score:  0.7966 (±0.0046)

🔹 KNN:
   Accuracy:  0.7779
   Precision: 0.5915
   Recall:    0.5201
   F1-Score:  0.5535
   AUC:       0.7941
   CV Score:  0.7528 (±0.0057)

🔹 NeuralNetwork:
   Accuracy:  0.8226
   Precision: 0.7030
   Recall:    0.5710
   F1-Score:  0.6302
   AUC:       0.8575
   CV Score:  0.7939 (±0.0052)
```

```python
# Hàm hỗ trợ: đảm bảo customer_data có đủ cột và đúng thứ tự
def prepare_input(customer_data, reference_columns):
    df_input = pd.DataFrame([customer_data])

    # Thêm cột thiếu với giá trị 0
    for col in reference_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Sắp xếp theo đúng thứ tự columns
    df_input = df_input[reference_columns]

    return df_input
```

```python
# Trung thành, khả năng ở lại cao
customer_1 = {
   'gender_Female': 0,
    'SeniorCitizen': 0,
    'Partner_Yes': 1,
    'tenure': 50,
    'MonthlyCharges': 65.0,
    'InternetService_DSL': 1,
    'Contract_Two year': 1,
    'PaymentMethod_Bank transfer (automatic)': 1
}

# Lớn tuổi, sống một mình, dịch vụ đắt 
customer_2 = {
    'gender_Female': 0,
    'SeniorCitizen': 1,
    'Partner_Yes': 0,
    'tenure': 2,
    'MonthlyCharges': 95.5,
    'InternetService_Fiber optic': 1,
    'Contract_Month-to-month': 1,
    'PaymentMethod_Mailed check': 1
}

# Mới, chi phí cao, hợp đồng ngắn hạn
customer_3 = {
    'gender_Female': 1,
    'SeniorCitizen': 1,
    'Partner_Yes': 0,
    'tenure': 1,
    'MonthlyCharges': 99.0,
    'InternetService_Fiber optic': 1,
    'Contract_Month-to-month': 1,
    'PaymentMethod_Electronic check': 1
}


multi_model.predict_all_customers(customer_1, X_train.columns, scaler)
multi_model.predict_all_customers(customer_2, X_train.columns, scaler)
multi_model.predict_all_customers(customer_3, X_train.columns, scaler)
```
## Kết quả
### ------------------------------🚀🚀🚀-----------------------------

🔍 Kết quả của mô hình LogisticRegression:
- -> 🛡️ Ở lại
- Xác suất churn: 0.0074

🔍 Kết quả của mô hình RandomForest:
- -> 🛡️ Ở lại
- Xác suất churn: 0.2260

🔍 Kết quả của mô hình SVM:
- -> 🛡️ Ở lại
- Xác suất churn: 0.1222

🔍 Kết quả của mô hình KNN:
- -> 🛡️ Ở lại
- Xác suất churn: 0.2569

🔍 Kết quả của mô hình NeuralNetwork:
- -> 🛡️ Ở lại
- Xác suất churn: 0.0000
### ------------------------------🚀🚀🚀-----------------------------

🔍 Kết quả của mô hình LogisticRegression:
- -> 🚶‍➡️ Rời đi
- Xc suất churn: 0.5162

🔍 Kết quả của mô hình RandomForest:
- -> 🚶‍➡️ Rời đi
- Xác suất churn: 0.6392

🔍 Kết quả của mô hình SVM:
- -> 🚶‍➡️ Rời đi
- Xác suất churn: 0.6521

🔍 Kết quả của mô hình KNN:
- -> 🚶‍➡️ Rời đi
- Xác suất churn: 0.8648

🔍 Kết quả của mô hình NeuralNetwork:
- -> 🛡️ Ở lại
- Xác suất churn: 0.0122

### 5.2.1 Linear Regression Analysis (Lecture 3)
```python
# Import thêm cho Linear Regression và Clustering
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score

# Lecture 3: Hồi quy tuyến tính - Phân tích mối quan hệ
print("\n📈 PHÂN TÍCH HỒI QUY TUYẾN TÍNH:")

# Hồi quy MonthlyCharges dựa trên tenure
lr_monthly = LinearRegression()
X_tenure = df_encoded[['tenure']].values
y_monthly = df_encoded['MonthlyCharges'].values

lr_monthly.fit(X_tenure, y_monthly)
y_pred_monthly = lr_monthly.predict(X_tenure)

r2_monthly = r2_score(y_monthly, y_pred_monthly)
mse_monthly = mean_squared_error(y_monthly, y_pred_monthly)

print(f"Hồi quy MonthlyCharges ~ tenure:")
print(f"R²: {r2_monthly:.4f}")
print(f"MSE: {mse_monthly:.4f}")
print(f"Hệ số hồi quy: {lr_monthly.coef_[0]:.4f}")
print(f"Intercept: {lr_monthly.intercept_:.4f}")

# Vẽ biểu đồ hồi quy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tenure, y_monthly, alpha=0.5, label='Dữ liệu thực')
plt.plot(X_tenure, y_pred_monthly, 'r-', label='Đường hồi quy')
plt.xlabel('Tenure (tháng)')
plt.ylabel('Monthly Charges')
plt.title(f'Hồi quy tuyến tính\nR² = {r2_monthly:.4f}')
plt.legend()

# Hồi quy TotalCharges dựa trên tenure + MonthlyCharges
lr_total = LinearRegression()
X_multi = df_encoded[['tenure', 'MonthlyCharges']].values
y_total = df_encoded['TotalCharges'].values

lr_total.fit(X_multi, y_total)
y_pred_total = lr_total.predict(X_multi)
r2_total = r2_score(y_total, y_pred_total)

print(f"\nHồi quy TotalCharges ~ tenure + MonthlyCharges:")
print(f"R²: {r2_total:.4f}")
print(f"Hệ số hồi quy: {lr_total.coef_}")

plt.subplot(1, 2, 2)
plt.scatter(y_total, y_pred_total, alpha=0.5)
plt.plot([y_total.min(), y_total.max()], [y_total.min(), y_total.max()], 'r--')
plt.xlabel('TotalCharges thực tế')
plt.ylabel('TotalCharges dự đoán')
plt.title(f'Dự đoán vs Thực tế\nR² = {r2_total:.4f}')

plt.tight_layout()
plt.show()
```

### 5.2.2 Clustering Analysis (Lecture 4+5)
```python
# Lecture 4+5: Phân cụm khách hàng
print("\n🎯 PHÂN TÍCH PHÂN CỤM KHÁCH HÀNG:")

# Chuẩn bị dữ liệu cho clustering (chỉ dùng numerical features)
cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_cluster = df_encoded[cluster_features].values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Tìm số cluster tối ưu bằng Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

# Vẽ biểu đồ Elbow và Silhouette
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow Method
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Số cụm (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Phương pháp Elbow')
axes[0].grid(True)

# Silhouette Score
axes[1].plot(K_range, silhouette_scores, 'ro-')
axes[1].set_xlabel('Số cụm (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Điểm Silhouette')
axes[1].grid(True)

# Chọn K tối ưu (K=3 hoặc K có silhouette score cao nhất)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Số cụm tối ưu: {optimal_k} (Silhouette Score: {max(silhouette_scores):.4f})")

# Thực hiện clustering với K tối ưu
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_optimal.fit_predict(X_cluster_scaled)

# Thêm cluster labels vào dataframe
df_encoded['Cluster'] = cluster_labels

# Phân tích clusters
axes[2].scatter(df_encoded['tenure'], df_encoded['MonthlyCharges'], 
               c=cluster_labels, cmap='viridis', alpha=0.6)
axes[2].set_xlabel('Tenure')
axes[2].set_ylabel('Monthly Charges')
axes[2].set_title(f'Phân cụm khách hàng (K={optimal_k})')

plt.tight_layout()
plt.show()

# Phân tích đặc điểm từng cluster
print("\n📊 ĐẶC ĐIỂM CÁC CLUSTER:")
for i in range(optimal_k):
    cluster_data = df_encoded[df_encoded['Cluster'] == i]
    churn_rate = cluster_data['Churn'].mean()
    
    print(f"\n🔸 Cluster {i} ({len(cluster_data)} khách hàng):")
    print(f"   Tỷ lệ Churn: {churn_rate:.2%}")
    print(f"   Tenure trung bình: {cluster_data['tenure'].mean():.1f} tháng")
    print(f"   MonthlyCharges trung bình: ${cluster_data['MonthlyCharges'].mean():.2f}")
    print(f"   TotalCharges trung bình: ${cluster_data['TotalCharges'].mean():.2f}")
```

### 5.3 Results Visualization
```python
# Chuyển đổi kết quả sang DataFrame để dễ phân tích
results_df = pd.DataFrame(results).T

print("📊 BẢNG SO SÁNH CHI TIẾT CÁC MÔ HÌNH:")
print(results_df.round(4))

# Tìm mô hình tốt nhất theo từng metric
print(f"\n🏆 MÔ HÌNH TỐT NHẤT THEO TỪNG METRIC:")
print(f"   Accuracy:  {results_df['accuracy'].idxmax()} ({results_df['accuracy'].max():.4f})")
print(f"   Precision: {results_df['precision'].idxmax()} ({results_df['precision'].max():.4f})")
print(f"   Recall:    {results_df['recall'].idxmax()} ({results_df['recall'].max():.4f})")
print(f"   F1-Score:  {results_df['f1_score'].idxmax()} ({results_df['f1_score'].max():.4f})")

# Biểu đồ so sánh hiệu suất
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# So sánh Accuracy
results_df['accuracy'].plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('So sánh Accuracy')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].set_ylim([results_df['accuracy'].min() - 0.05, 1.0])

# So sánh Precision
results_df['precision'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('So sánh Precision')
axes[0,1].set_ylabel('Precision')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].set_ylim([results_df['precision'].min() - 0.05, 1.0])

# So sánh Recall
results_df['recall'].plot(kind='bar', ax=axes[0,2], color='lightcoral')
axes[0,2].set_title('So sánh Recall')
axes[0,2].set_ylabel('Recall')
axes[0,2].tick_params(axis='x', rotation=45)
axes[0,2].set_ylim([results_df['recall'].min() - 0.05, 1.0])

# So sánh F1-Score
results_df['f1_score'].plot(kind='bar', ax=axes[1,0], color='gold')
axes[1,0].set_title('So sánh F1-Score')
axes[1,0].set_ylabel('F1-Score')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].set_ylim([results_df['f1_score'].min() - 0.05, 1.0])

# So sánh AUC (chỉ cho models có AUC)
auc_data = results_df[results_df['auc'].notna()]['auc']
if len(auc_data) > 0:
    auc_data.plot(kind='bar', ax=axes[1,1], color='plum')
    axes[1,1].set_title('So sánh AUC Score')
    axes[1,1].set_ylabel('AUC')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylim([auc_data.min() - 0.05, 1.0])

# Cross Validation Scores với error bars
cv_data = results_df[results_df['cv_mean'].notna()]
if len(cv_data) > 0:
    cv_means = cv_data['cv_mean']
    cv_stds = cv_data['cv_std']
    x_pos = range(len(cv_means))
    axes[1,2].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color='orange', alpha=0.7)
    axes[1,2].set_title('Cross Validation Scores')
    axes[1,2].set_ylabel('CV Accuracy')
    axes[1,2].set_xticks(x_pos)
    axes[1,2].set_xticklabels(cv_means.index, rotation=45)

plt.tight_layout()
plt.show()

# Ma trận nhầm lẫn cho top 3 models theo accuracy
print(f"\n🎯 MA TRẬN NHẦM LẪN CHO TOP 3 MÔ HÌNH:")
top_3_models = results_df.nlargest(3, 'accuracy').index

fig, axes = plt.subplots(1, min(3, len(top_3_models)), figsize=(15, 4))
if len(top_3_models) == 1:
    axes = [axes]

for i, model_name in enumerate(top_3_models[:3]):
    # Lấy predictions từ model
    model = multi_model.models[model_name]
    y_pred = model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'{model_name}\nAccuracy: {results[model_name]["accuracy"]:.3f}')
    axes[i].set_xlabel('Dự đoán')
    axes[i].set_ylabel('Thực tế')

plt.tight_layout()
plt.show()

# Biểu đồ radar cho so sánh tổng quan
import numpy as np

def create_radar_chart(models_data, metrics=['accuracy', 'precision', 'recall', 'f1_score']):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Số lượng metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    # Vẽ cho từng model
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, data) in enumerate(models_data.items()):
        values = [data[metric] for metric in metrics]
        values += values[:1]  # Đóng vòng tròn
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Thiết lập labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.title() for metric in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('So sánh tổng quan các mô hình', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.show()

# Tạo radar chart
print(f"\n📈 BIỂU ĐỒ RADAR SO SÁNH TỔNG QUAN:")
create_radar_chart(results)
```

### 5.4 Market Basket Analysis (Lecture 12)
```python
# Lecture 12: Khai phá tập mục thường xuyên và các luật kết hợp
print("\n🛒 PHÂN TÍCH TẬP MỤC THƯỜNG XUYÊN (MARKET BASKET ANALYSIS):")

# Cài đặt thư viện mlxtend nếu chưa có
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    print("Cài đặt mlxtend...")
    import subprocess
    subprocess.run(["pip", "install", "mlxtend"], check=True)
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

# Tạo dữ liệu giao dịch từ các dịch vụ
service_cols = ['PhoneService', 'MultipleLines', 'InternetService_DSL', 'InternetService_Fiber optic',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies']

# Lọc các cột dịch vụ có trong dữ liệu
available_service_cols = [col for col in service_cols if col in df_encoded.columns]
print(f"Các dịch vụ phân tích: {available_service_cols}")

# Tạo transactions (chỉ lấy các dịch vụ được sử dụng = 1)
transactions = []
for _, row in df_encoded.iterrows():
    transaction = []
    for col in available_service_cols:
        if col in df_encoded.columns and row[col] == 1:
            transaction.append(col)
    
    # Thêm thông tin churn
    if row['Churn'] == 1:
        transaction.append('Churn_Yes')
    else:
        transaction.append('Churn_No')
    
    # Thêm thông tin contract
    if 'Contract_One year' in df_encoded.columns and row['Contract_One year'] == 1:
        transaction.append('Contract_OneYear')
    elif 'Contract_Two year' in df_encoded.columns and row['Contract_Two year'] == 1:
        transaction.append('Contract_TwoYear')
    else:
        transaction.append('Contract_MonthToMonth')
    
    transactions.append(transaction)

# Chuyển đổi thành format cho apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Số lượng transactions: {len(df_transactions)}")
print(f"Số lượng items: {len(te.columns_)}")
print(f"Items: {list(te.columns_)}")

# Tìm frequent itemsets
frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(f"\n📊 SỐ LƯỢNG FREQUENT ITEMSETS:")
itemset_counts = frequent_itemsets['length'].value_counts().sort_index()
print(itemset_counts)

# Hiển thị top frequent itemsets
print(f"\n🔝 TOP 10 FREQUENT ITEMSETS:")
top_itemsets = frequent_itemsets.nlargest(10, 'support')
for idx, row in top_itemsets.iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"Support: {row['support']:.3f} | Items: {items}")

# Tạo association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

if len(rules) > 0:
    # Sắp xếp theo lift
    rules_sorted = rules.sort_values('lift', ascending=False)
    
    print(f"\n🔗 TOP 10 ASSOCIATION RULES (sắp xếp theo Lift):")
    for idx, rule in rules_sorted.head(10).iterrows():
        antecedent = ', '.join(list(rule['antecedents']))
        consequent = ', '.join(list(rule['consequents']))
        print(f"Rule: {antecedent} → {consequent}")
        print(f"   Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
        print()
    
    # Phân tích rules liên quan đến Churn
    churn_rules = rules_sorted[rules_sorted['consequents'].astype(str).str.contains('Churn')]
    
    if len(churn_rules) > 0:
        print(f"\n⚠️  TOP 5 RULES DẪN ĐẾN CHURN:")
        for idx, rule in churn_rules.head(5).iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            print(f"Rule: {antecedent} → {consequent}")
            print(f"   Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
            print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'])
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence')
    plt.colorbar(label='Lift')
    
    plt.subplot(1, 3, 2)
    plt.hist(rules['lift'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.title('Phân bố Lift')
    
    plt.subplot(1, 3, 3)
    itemset_counts.plot(kind='bar', color='lightgreen')
    plt.xlabel('Độ dài Itemset')
    plt.ylabel('Số lượng')
    plt.title('Phân bố độ dài Itemset')
    
    plt.tight_layout()
    plt.show()
else:
    print("Không tìm thấy association rules với threshold hiện tại.")
```

## 🎯 Bước 6: Feature Selection Performance

### 6.1 Test với Different Feature Sets
```python
# Kiểm tra hiệu suất với các bộ đặc trưng khác nhau
feature_sets = {
    'Top 5 Univariate': select_features_univariate(X, y, 5)[0],
    'Top 10 Univariate': select_features_univariate(X, y, 10)[0],
    'Top 15 Univariate': select_features_univariate(X, y, 15)[0],
    'Top 5 RF': select_features_rf(X, y, 5)[0],
    'Top 10 RF': select_features_rf(X, y, 10)[0],
    'Top 15 RF': select_features_rf(X, y, 15)[0],
}

# Hiệu suất với các tập con đặc trưng
subset_results = {}

for subset_name, features in feature_sets.items():
    print(f"\n🔍 Đang kiểm tra {subset_name} ({len(features)} đặc trưng)...")
    
    # Tạo tập con dữ liệu
    X_subset = X[features]
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_subset, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa
    X_train_sub_scaled = scaler.fit_transform(X_train_sub)
    X_test_sub_scaled = scaler.transform(X_test_sub)
    
    # Huấn luyện mô hình tốt nhất (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sub_scaled, y_train_sub)
    
    # Đánh giá
    y_pred_sub = model.predict(X_test_sub_scaled)
    accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)
    
    subset_results[subset_name] = {
        'n_features': len(features),
        'accuracy': accuracy_sub,
        'features': features
    }
    
    print(f"   Độ chính xác: {accuracy_sub:.4f} với {len(features)} đặc trưng")

# Biểu đồ kết quả lựa chọn đặc trưng
subset_df = pd.DataFrame(subset_results).T
subset_df = subset_df.sort_values('accuracy', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(subset_df)), subset_df['accuracy'], color='lightcoral')
plt.xlabel('Phương pháp lựa chọn đặc trưng')
plt.ylabel('Độ chính xác')
plt.title('Hiệu suất theo phương pháp lựa chọn đặc trưng')
plt.xticks(range(len(subset_df)), subset_df.index, rotation=45)

# Thêm nhãn giá trị trên cột
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}\n({subset_df.iloc[i]["n_features"]} đặc trưng)',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n🏆 KẾT QUẢ CHỌN ĐẶC TRƯNG TỐT NHẤT:")
print(subset_df.sort_values('accuracy', ascending=False))
```

## 💾 Bước 7: Save Results

### 7.1 Export Results
```python
# Lưu kết quả ra CSV
results_df.to_csv('model_comparison_results.csv')
subset_df.to_csv('feature_selection_results.csv')

# Tải xuống các file
from google.colab import files
files.download('model_comparison_results.csv')
files.download('feature_selection_results.csv')

print("✅ Kết quả đã được lưu và tải xuống!")
```

## 🎉 Hoàn thành!

### �� Summary
```python
# Tóm tắt cải tiến - chọn mô hình theo business metrics
def select_best_model_for_churn(results_dict):
    """
    Chọn mô hình tốt nhất cho bài toán churn prediction
    Ưu tiên: Recall > AUC > F1-Score > Accuracy
    """
    scores = {}
    for model, metrics in results_dict.items():
        # Weighted business score for churn prediction
        business_score = (
            metrics['recall'] * 0.35 +      # Quan trọng nhất: phát hiện churn
            metrics['auc'] * 0.25 +         # Khả năng phân biệt
            metrics['f1_score'] * 0.25 +    # Cân bằng precision/recall  
            metrics['accuracy'] * 0.15      # Độ chính xác tổng thể
        )
        scores[model] = business_score
    
    best_model = max(scores, key=scores.get)
    return best_model, scores

# Sử dụng function
best_model, business_scores = select_best_model_for_churn(results)

print("🎯 TÓM TẮT DỰ ÁN")
print("=" * 50)
print(f"📊 Dữ liệu: {df.shape[0]} khách hàng, {df.shape[1]} đặc trưng")
print(f"🎯 Mục tiêu: Customer Churn ({(y.sum() / len(y) * 100):.1f}% tỷ lệ churn)")
print(f"🔧 Đặc trưng sau tiền xử lý: {X.shape[1]}")
print(f"🏆 Mô hình tốt nhất (Accuracy): {results_df['accuracy'].idxmax()} ({results_df['accuracy'].max():.3f})")
print(f"🎯 Mô hình tốt nhất (Business): {best_model} (Score: {business_scores[best_model]:.3f})")
print(f"📊 Bộ đặc trưng tốt nhất: {subset_df.index[0]} ({subset_df.iloc[0]['accuracy']:.3f})")

print(f"\n📈 BUSINESS SCORES CHO TẤT CẢ MÔ HÌNH:")
for model, score in sorted(business_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model}: {score:.3f}")

print("\n✅ Phân tích hoàn thành thành công!")
```