# Bộ Dữ Liệu Telco Customer Churn

## 📊 Tổng quan

Đây là bộ dữ liệu **Telco Customer Churn** (Tỷ lệ rời bỏ khách hàng viễn thông), một dataset phổ biến trong lĩnh vực phân tích dữ liệu và machine learning. Bộ dữ liệu chứa thông tin về **7,043 khách hàng** của một công ty viễn thông.

## 🎯 Mục đích sử dụng

Dataset này được thiết kế để:
- **Dự đoán** khách hàng có khả năng rời bỏ dịch vụ (churn) hay không
- **Phân tích** các yếu tố ảnh hưởng đến quyết định của khách hàng
- **Xây dựng mô hình** machine learning cho bài toán classification
- **Nghiên cứu** hành vi khách hàng trong ngành viễn thông

## 📁 Cấu trúc dữ liệu

Bộ dữ liệu gồm **21 cột** với các thông tin sau:

| Tên cột | Mô tả |
|---------|-------|
| `customerID` | Mã định danh khách hàng |
| `gender` | Giới tính (Male/Female) |
| `SeniorCitizen` | Người cao tuổi (0/1) |
| `Partner` | Có người bạn đời (Yes/No) |
| `Dependents` | Có người phụ thuộc (Yes/No) |
| `tenure` | Thời gian sử dụng dịch vụ (tháng) |
| `PhoneService` | Dịch vụ điện thoại |
| `MultipleLines` | Nhiều đường dây |
| `InternetService` | Dịch vụ Internet (DSL/Fiber optic/No) |
| `OnlineSecurity` | Bảo mật trực tuyến |
| `OnlineBackup` | Sao lưu trực tuyến |
| `DeviceProtection` | Bảo vệ thiết bị |
| `TechSupport` | Hỗ trợ kỹ thuật |
| `StreamingTV` | Dịch vụ TV streaming |
| `StreamingMovies` | Dịch vụ phim streaming |
| `Contract` | Loại hợp đồng |
| `PaperlessBilling` | Hóa đơn không giấy |
| `PaymentMethod` | Phương thức thanh toán |
| `MonthlyCharges` | Phí hàng tháng ($) |
| `TotalCharges` | Tổng phí đã thanh toán ($) |
| `Churn` | Khách hàng có rời bỏ dịch vụ hay không |


## Chi tiết
### 👤 Thông tin cá nhân khách hàng
| Cột | Mô tả | Kiểu dữ liệu |
|-----|-------|--------------|
| `customerID` | Mã định danh khách hàng | String |
| `gender` | Giới tính (Male/Female) | Categorical |
| `SeniorCitizen` | Người cao tuổi (0/1) | Binary |
| `Partner` | Có người bạn đời (Yes/No) | Categorical |
| `Dependents` | Có người phụ thuộc (Yes/No) | Categorical |
| `tenure` | Thời gian sử dụng dịch vụ (tháng) | Numerical |

### 📞 Thông tin dịch vụ
| Cột | Mô tả | Kiểu dữ liệu |
|-----|-------|--------------|
| `PhoneService` | Dịch vụ điện thoại | Categorical |
| `MultipleLines` | Nhiều đường dây | Categorical |
| `InternetService` | Dịch vụ Internet (DSL/Fiber optic/No) | Categorical |
| `OnlineSecurity` | Bảo mật trực tuyến | Categorical |
| `OnlineBackup` | Sao lưu trực tuyến | Categorical |
| `DeviceProtection` | Bảo vệ thiết bị | Categorical |
| `TechSupport` | Hỗ trợ kỹ thuật | Categorical |
| `StreamingTV` | Dịch vụ TV streaming | Categorical |
| `StreamingMovies` | Dịch vụ phim streaming | Categorical |

### 💳 Thông tin hợp đồng và thanh toán
| Cột | Mô tả | Kiểu dữ liệu |
|-----|-------|--------------|
| `Contract` | Loại hợp đồng | Categorical |
| `PaperlessBilling` | Hóa đơn không giấy | Categorical |
| `PaymentMethod` | Phương thức thanh toán | Categorical |
| `MonthlyCharges` | Phí hàng tháng ($) | Numerical |
| `TotalCharges` | Tổng phí đã thanh toán ($) | Numerical |

### 🎯 Biến mục tiêu
| Cột | Mô tả | Kiểu dữ liệu |
|-----|-------|--------------|
| `Churn` | Khách hàng có rời bỏ dịch vụ hay không | Binary (Yes/No) |

## 📈 Đặc điểm dữ liệu

- **Kích thước**: 7,043 records × 21 features
- **Dung lượng**: ~955KB
- **Loại bài toán**: Binary Classification (Churn: Yes/No)
- **Dữ liệu có cấu trúc**: Mix giữa categorical và numerical features
- **Chất lượng**: Dữ liệu tương đối sạch, ít missing values

## 🔬 Ứng dụng thực tế

Bộ dữ liệu này rất phù hợp cho:

### 📚 Học tập và Nghiên cứu
- Thực hành các thuật toán machine learning
- Exploratory Data Analysis (EDA)
- Feature engineering và data preprocessing
- Model evaluation và comparison

### 💼 Business Applications
- Customer retention strategies
- Predictive analytics
- Customer segmentation
- Revenue optimization

### 🤖 Machine Learning Projects
- Binary classification models
- Ensemble methods
- Deep learning applications
- AutoML experiments

## 🛠️ Kỹ thuật phân tích thường được áp dụng

### Classification Algorithms
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks

### Data Processing
- **Encoding categorical variables**: One-hot encoding, Label encoding
- **Scaling numerical features**: StandardScaler, MinMaxScaler
- **Feature selection**: Correlation analysis, Feature importance
- **Handling imbalanced data**: SMOTE, undersampling, oversampling

### Data Visualization
- Distribution analysis
- Correlation heatmaps
- Churn patterns visualization
- Customer segmentation plots

## 📊 Thông tin thống kê cơ bản

```python
# Ví dụ code để load và khám phá dữ liệu
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('telco-customer-churn.csv')

# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)}")
```

## 🚀 Bắt đầu nhanh

1. **Load dữ liệu**
  ```python
  import pandas as pd
  df = pd.read_csv('telco-customer-churn.csv')
  ```

2. **Khám phá dữ liệu**
  ```python
  df.info()
  df.describe()
  df.head()
  ```

3. **Phân tích Churn**
  ```python
  churn_rate = df['Churn'].value_counts(normalize=True)
  print(f"Churn rate: {churn_rate['Yes']:.2%}")
  ```

## 📝 Lưu ý quan trọng

- **TotalCharges** có thể chứa giá trị string " " cần được xử lý
- Một số features có giá trị "No internet service" hoặc "No phone service"
- Cần cân nhắc class imbalance trong bài toán churn prediction
- Feature engineering có thể cải thiện đáng kể performance

## 🎯 Mục tiêu học tập

Bộ dữ liệu này giúp bạn:
- Hiểu rõ quy trình phân tích dữ liệu end-to-end
- Thực hành các kỹ thuật preprocessing
- Áp dụng nhiều thuật toán machine learning
- Đánh giá và so sánh hiệu suất mô hình
- Hiểu về business context trong customer analytics

---

**Happy Learning! 🎉**

*Dataset này là nền tảng tuyệt vời để bắt đầu hành trình machine learning trong lĩnh vực customer analytics và churn prediction.*
