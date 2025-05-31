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
# Nếu có link download trực tiếp
url = "https://raw.githubusercontent.com/tanmaiii/may_hoc_ung_dung/refs/heads/main/telco-customer-churn.csv"
df = pd.read_csv(url)
```

## 📊 Bước 2: Notebook hoàn chỉnh

### 2.1 Setup và Import Libraries
```python
# Cài đặt thêm packages nếu cần
!pip install plotly seaborn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Tắt warnings
import warnings
warnings.filterwarnings('ignore')

# Cài đặt style cho plots
plt.style.use('default')
sns.set_palette("husl")
```

### 2.2 Load và Explore Data (Load dữ liệu)
```python
# Load data
df = pd.read_csv('telco-customer-churn.csv')  # Hoặc đường dẫn file bạn upload

print("🎯 TỔNG QUAN DỮ LIỆU")
print(f"Kích thước: {df.shape}")
print(f"Giá trị thiếu: {df.isnull().sum().sum()}")
print("\n📊 5 dòng đầu tiên:")
df.head()
```

### 2.3 Data Analysis và Visualization (Phân tích dữ liệu và trực quan hóa)
```python
# Basic info
print("📈 THÔNG TIN DỮ LIỆU")
df.info()

print("\n📊 THỐNG KÊ MÔ TẢ")
df.describe()

# Churn distribution
print("\n🎯 PHÂN BỐ CHURN")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Tỷ lệ khách hàng rời bỏ dịch vụ (Churn rate): {churn_counts[1] / len(df) * 100:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Biểu đồ pie char
axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%')
axes[0,0].set_title('Tỷ lệ khách hàng rời bỏ (Churn)')

# Biểu đồ Histogram
axes[0,1].hist(df['tenure'], bins=30, alpha=0.7)
axes[0,1].set_title('Phân bố thời gian sử dụng')
axes[0,1].set_xlabel('Số tháng sử dụng (tenure)')

# Biểu đồ Boxplot
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0])
axes[1,0].set_title('Chi phí hàng tháng theo Churn')

# Biểu đồ Bar chart
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
# Tạo copy để xử lý
df_clean = df.copy()

print("🧹 LÀM SẠCH DỮ LIỆU...")

# Xử lý TotalCharges (có giá trị ' ')
print(f"Giá trị ' ' trong TotalCharges: {(df_clean['TotalCharges'] == ' ').sum()}")
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Fill missing values
df_clean['TotalCharges'].fillna(0, inplace=True)

# Remove customerID
df_clean = df_clean.drop('customerID', axis=1)

print(f"✅ Kích thước sau làm sạch: {df_clean.shape}")
print(f"Giá trị thiếu sau làm sạch: {df_clean.isnull().sum().sum()}")
```

### 3.2 Feature Engineering 
```python
print("🔧 TẠO ĐẶC TRƯNG MỚI...")

# Binary encoding cho Yes/No columns
binary_cols = []
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' and col != 'Churn':
        unique_vals = df_clean[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            binary_cols.append(col)
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

print(f"Các cột đã mã hóa nhị phân: {binary_cols}")

# One-hot encoding cho nominal variables
nominal_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
df_encoded = pd.get_dummies(df_clean, columns=nominal_cols, drop_first=True)

# Encode target variable
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create new features
df_encoded['AvgChargePerTenure'] = np.where(
    df_encoded['tenure'] > 0,
    df_encoded['TotalCharges'] / df_encoded['tenure'],
    df_encoded['MonthlyCharges']
)

# Tenure groups - tạo nhóm theo thời gian sử dụng
try:
    df_encoded['TenureGroup'] = pd.cut(
        df_encoded['tenure'],
        bins=[0, 12, 36, 60, float('inf')],
        labels=['0-12m', '13-36m', '37-60m', '60m+']
    )
    # Encode categorical labels thành số
    df_encoded['TenureGroup'] = df_encoded['TenureGroup'].cat.codes
except Exception as e:
    print(f"Lỗi khi tạo TenureGroup: {e}")
    # Fallback: tạo nhóm đơn giản
    df_encoded['TenureGroup'] = pd.cut(
        df_encoded['tenure'],
        bins=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

print(f"✅ Kích thước cuối cùng: {df_encoded.shape}")
df_encoded.head()
```

## 🎯 Bước 4: Feature Selection

### 4.1 Correlation Analysis
```python
# Correlation heatmap
plt.figure(figsize=(20, 16))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Ma trận tương quan đặc trưng')
plt.show()

# Top correlations with target
target_corr = correlation_matrix['Churn'].abs().sort_values(ascending=False)
print("🎯 TOP 15 ĐẶC TRƯNG TƯƠNG QUAN VỚI CHURN:")
print(target_corr.head(15))
```

### 4.2 Feature Selection Methods
```python
# Prepare data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Kích thước đặc trưng: {X.shape}")
print(f"Phân bố target: {y.value_counts().to_dict()}")

# Method 1: SelectKBest
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

# Method 2: Random Forest Feature Importance
def select_features_rf(X, y, k=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance.head(k)['feature'].tolist()
    return selected_features, feature_importance

# Apply feature selection
for k in [5, 10, 15]:
    print(f"\n🔍 TOP {k} ĐẶC TRƯNG:")
    
    # Univariate selection
    features_uni, scores_uni = select_features_univariate(X, y, k)
    print(f"Phương pháp Univariate: {features_uni}")
    
    # Random Forest
    features_rf, scores_rf = select_features_rf(X, y, k)
    print(f"Random Forest: {features_rf}")
```

## 🤖 Bước 5: Model Training và Evaluation

### 5.1 Data Splitting và Scaling
```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Tập huấn luyện: {X_train_scaled.shape}")
print(f"Tập kiểm tra: {X_test_scaled.shape}")
print(f"Phân bố target trong tập train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
```

### 5.2 Model Training
```python
# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\n🔄 Đang huấn luyện {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'accuracy': accuracy,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"   Độ chính xác: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}" if auc else "   AUC: Không có")
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

### 5.3 Results Visualization
```python
# Results comparison
results_df = pd.DataFrame({
    name: {
        'Test Accuracy': results[name]['accuracy'],
        'AUC Score': results[name]['auc'] if results[name]['auc'] else 0,
        'CV Mean': results[name]['cv_mean'],
        'CV Std': results[name]['cv_std']
    } for name in results.keys()
}).T

print("📊 SO SÁNH MÔ HÌNH:")
print(results_df.round(4))

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy comparison
results_df['Test Accuracy'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('So sánh độ chính xác kiểm tra')
axes[0].set_ylabel('Độ chính xác')
axes[0].tick_params(axis='x', rotation=45)

# AUC comparison
results_df['AUC Score'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('So sánh điểm AUC')
axes[1].set_ylabel('AUC')
axes[1].tick_params(axis='x', rotation=45)

# CV scores with error bars
cv_means = results_df['CV Mean']
cv_stds = results_df['CV Std']
axes[2].bar(range(len(cv_means)), cv_means, yerr=cv_stds, capsize=5, color='orange', alpha=0.7)
axes[2].set_title('Điểm Cross Validation')
axes[2].set_ylabel('CV Accuracy')
axes[2].set_xticks(range(len(cv_means)))
axes[2].set_xticklabels(cv_means.index, rotation=45)

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'{name}\nĐộ chính xác: {result["accuracy"]:.3f}')
    axes[i].set_xlabel('Dự đoán')
    axes[i].set_ylabel('Thực tế')

plt.tight_layout()
plt.show()
```

## 🎯 Bước 6: Feature Selection Performance

### 6.1 Test với Different Feature Sets
```python
# Test performance với different feature sets
feature_sets = {
    'Top 5 Univariate': select_features_univariate(X, y, 5)[0],
    'Top 10 Univariate': select_features_univariate(X, y, 10)[0],
    'Top 15 Univariate': select_features_univariate(X, y, 15)[0],
    'Top 5 RF': select_features_rf(X, y, 5)[0],
    'Top 10 RF': select_features_rf(X, y, 10)[0],
    'Top 15 RF': select_features_rf(X, y, 15)[0],
}

# Performance với feature subsets
subset_results = {}

for subset_name, features in feature_sets.items():
    print(f"\n🔍 Đang kiểm tra {subset_name} ({len(features)} đặc trưng)...")
    
    # Subset data
    X_subset = X[features]
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_subset, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    X_train_sub_scaled = scaler.fit_transform(X_train_sub)
    X_test_sub_scaled = scaler.transform(X_test_sub)
    
    # Train best model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sub_scaled, y_train_sub)
    
    # Evaluate
    y_pred_sub = model.predict(X_test_sub_scaled)
    accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)
    
    subset_results[subset_name] = {
        'n_features': len(features),
        'accuracy': accuracy_sub,
        'features': features
    }
    
    print(f"   Độ chính xác: {accuracy_sub:.4f} với {len(features)} đặc trưng")

# Plot feature selection results
subset_df = pd.DataFrame(subset_results).T
subset_df = subset_df.sort_values('accuracy', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(subset_df)), subset_df['accuracy'], color='lightcoral')
plt.xlabel('Phương pháp lựa chọn đặc trưng')
plt.ylabel('Độ chính xác')
plt.title('Hiệu suất theo phương pháp lựa chọn đặc trưng')
plt.xticks(range(len(subset_df)), subset_df.index, rotation=45)

# Add value labels on bars
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
# Save results to CSV
results_df.to_csv('model_comparison_results.csv')
subset_df.to_csv('feature_selection_results.csv')

# Download files
from google.colab import files
files.download('model_comparison_results.csv')
files.download('feature_selection_results.csv')

print("✅ Kết quả đã được lưu và tải xuống!")
```

## 🎉 Hoàn thành!

### 📊 Summary
```python
print("🎯 TÓM TẮT DỰ ÁN")
print("=" * 50)
print(f"📊 Dữ liệu: {df.shape[0]} khách hàng, {df.shape[1]} đặc trưng")
print(f"🎯 Mục tiêu: Customer Churn ({(y.sum() / len(y) * 100):.1f}% tỷ lệ churn)")
print(f"🔧 Đặc trưng sau tiền xử lý: {X.shape[1]}")
print(f"🏆 Mô hình tốt nhất: {results_df['Test Accuracy'].idxmax()} ({results_df['Test Accuracy'].max():.3f})")
print(f"🎯 Bộ đặc trưng tốt nhất: {subset_df.index[0]} ({subset_df.iloc[0]['accuracy']:.3f})")
print("\n✅ Phân tích hoàn thành thành công!")
```

## 🚀 Tips cho Google Colab

### Tối ưu Performance:
- Sử dụng GPU: **Runtime > Change runtime type > GPU**
- Disconnect sau khi xong: **Runtime > Disconnect**
- Restart nếu memory đầy: **Runtime > Restart runtime**

### Lưu trữ:
- Mount Google Drive để lưu permanent
- Download results quan trọng về máy
- Copy notebook vào Drive để backup

### Troubleshooting:
- Nếu bị disconnect: Re-run từ đầu
- Nếu thiếu package: `!pip install package_name`
- Nếu lỗi memory: Giảm data size hoặc dùng sampling

---

**🎉 Chúc bạn thành công với project Machine Learning!** 