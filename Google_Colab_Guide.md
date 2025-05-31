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
4. Đổi tên: **Telco_Customer_Churn_ML_Project**

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

### 2.2 Load và Explore Data
```python
# Load data
df = pd.read_csv('telco-customer-churn.csv')  # Hoặc đường dẫn file bạn upload

print("🎯 DATASET OVERVIEW")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\n📊 First 5 rows:")
df.head()
```

### 2.3 Data Analysis và Visualization
```python
# Basic info
print("📈 DATA INFO")
df.info()

print("\n📊 STATISTICAL SUMMARY")
df.describe()

# Churn distribution
print("\n🎯 CHURN DISTRIBUTION")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Churn rate: {churn_counts['Yes'] / len(df) * 100:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Churn distribution
axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%')
axes[0,0].set_title('Churn Distribution')

# Tenure distribution
axes[0,1].hist(df['tenure'], bins=30, alpha=0.7)
axes[0,1].set_title('Tenure Distribution')
axes[0,1].set_xlabel('Tenure (months)')

# Monthly charges by churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0])
axes[1,0].set_title('Monthly Charges by Churn')

# Contract type by churn
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Contract Type vs Churn')
axes[1,1].legend(title='Churn')

plt.tight_layout()
plt.show()
```

## 🔄 Bước 3: Data Preprocessing

### 3.1 Data Cleaning
```python
# Tạo copy để xử lý
df_clean = df.copy()

print("🧹 DATA CLEANING...")

# Xử lý TotalCharges (có giá trị ' ')
print(f"TotalCharges ' ' values: {(df_clean['TotalCharges'] == ' ').sum()}")
df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Fill missing values
df_clean['TotalCharges'].fillna(0, inplace=True)

# Remove customerID
df_clean = df_clean.drop('customerID', axis=1)

print(f"✅ Cleaned shape: {df_clean.shape}")
print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
```

### 3.2 Feature Engineering
```python
print("🔧 FEATURE ENGINEERING...")

# Binary encoding cho Yes/No columns
binary_cols = []
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' and col != 'Churn':
        unique_vals = df_clean[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            binary_cols.append(col)
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

print(f"Binary encoded columns: {binary_cols}")

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

# Tenure groups
df_encoded['TenureGroup'] = pd.cut(
    df_encoded['tenure'],
    bins=[0, 12, 36, 60, float('inf')],
    labels=[0, 1, 2, 3]
)
df_encoded['TenureGroup'] = df_encoded['TenureGroup'].astype(int)

print(f"✅ Final encoded shape: {df_encoded.shape}")
df_encoded.head()
```

## 🎯 Bước 4: Feature Selection

### 4.1 Correlation Analysis
```python
# Correlation heatmap
plt.figure(figsize=(20, 16))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Top correlations with target
target_corr = correlation_matrix['Churn'].abs().sort_values(ascending=False)
print("🎯 TOP 15 FEATURES CORRELATED WITH CHURN:")
print(target_corr.head(15))
```

### 4.2 Feature Selection Methods
```python
# Prepare data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

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
    print(f"\n🔍 TOP {k} FEATURES:")
    
    # Univariate selection
    features_uni, scores_uni = select_features_univariate(X, y, k)
    print(f"Univariate: {features_uni}")
    
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

print(f"Train set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"Train target distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")
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
    print(f"\n🔄 Training {name}...")
    
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
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}" if auc else "   AUC: N/A")
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

print("📊 MODEL COMPARISON:")
print(results_df.round(4))

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy comparison
results_df['Test Accuracy'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Test Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=45)

# AUC comparison
results_df['AUC Score'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('AUC Score Comparison')
axes[1].set_ylabel('AUC')
axes[1].tick_params(axis='x', rotation=45)

# CV scores with error bars
cv_means = results_df['CV Mean']
cv_stds = results_df['CV Std']
axes[2].bar(range(len(cv_means)), cv_means, yerr=cv_stds, capsize=5, color='orange', alpha=0.7)
axes[2].set_title('Cross Validation Scores')
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
    axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

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
    print(f"\n🔍 Testing {subset_name} ({len(features)} features)...")
    
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
    
    print(f"   Accuracy: {accuracy_sub:.4f} with {len(features)} features")

# Plot feature selection results
subset_df = pd.DataFrame(subset_results).T
subset_df = subset_df.sort_values('accuracy', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(subset_df)), subset_df['accuracy'], color='lightcoral')
plt.xlabel('Feature Selection Method')
plt.ylabel('Accuracy')
plt.title('Performance by Feature Selection Method')
plt.xticks(range(len(subset_df)), subset_df.index, rotation=45)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.3f}\n({subset_df.iloc[i]["n_features"]} features)',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n🏆 BEST FEATURE SELECTION RESULTS:")
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

print("✅ Results saved and downloaded!")
```

## 🎉 Hoàn thành!

### 📊 Summary
```python
print("🎯 PROJECT SUMMARY")
print("=" * 50)
print(f"📊 Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(f"🎯 Target: Customer Churn ({(y.sum() / len(y) * 100):.1f}% churn rate)")
print(f"🔧 Features after preprocessing: {X.shape[1]}")
print(f"🏆 Best model: {results_df['Test Accuracy'].idxmax()} ({results_df['Test Accuracy'].max():.3f})")
print(f"🎯 Best feature set: {subset_df.index[0]} ({subset_df.iloc[0]['accuracy']:.3f})")
print("\n✅ Analysis completed successfully!")
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