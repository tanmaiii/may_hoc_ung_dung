# ⚡ GOOGLE COLAB - QUICK START

## 🚀 Bắt đầu nhanh trong 3 phút

### Bước 1: Mở Google Colab
1. Truy cập: [colab.research.google.com](https://colab.research.google.com/)
2. Đăng nhập Google account
3. **File > New notebook** hoặc upload file `Telco_Churn_Colab.ipynb`

### Bước 2: Upload dữ liệu
```python
from google.colab import files
uploaded = files.upload()
# Chọn file telco-customer-churn.csv
```

### Bước 3: Chạy toàn bộ project
- **Runtime > Run all** hoặc Ctrl+F9
- Hoặc chạy từng cell bằng Shift+Enter

## 📊 Kết quả mong đợi

Sau khi chạy xong bạn sẽ có:
- ✅ **Accuracy ~80-85%** cho các models
- ✅ **Feature selection** với top 5, 10, 15 features
- ✅ **Model comparison** (Logistic, Random Forest, SVM)
- ✅ **Visualization** đầy đủ
- ✅ **Results files** để download

## 🎯 Highlight Features

### Data Processing:
- ✅ Missing values handling
- ✅ Feature encoding (binary, one-hot)
- ✅ New feature creation
- ✅ Data scaling

### Feature Selection:
- ✅ Univariate selection (SelectKBest)
- ✅ Random Forest importance
- ✅ Performance comparison
- ✅ Top 5/10/15 features analysis

### Machine Learning:
- ✅ 3 different algorithms
- ✅ Cross-validation
- ✅ Hyperparameter optimization
- ✅ Comprehensive metrics

### Visualization:
- ✅ EDA plots
- ✅ Model comparison charts
- ✅ Feature importance plots
- ✅ Confusion matrices

## 🔧 Tối ưu Colab

### Sử dụng GPU (tùy chọn):
- **Runtime > Change runtime type > GPU**

### Lưu trữ persistent:
```python
from google.colab import drive
drive.mount('/content/drive')
# Save files to Google Drive
```

### Download kết quả:
```python
from google.colab import files
files.download('results.csv')
```

## 🆘 Troubleshooting

**Lỗi thường gặp:**

1. **File not found**: Kiểm tra đã upload đúng file CSV chưa
2. **Memory error**: Restart runtime và chạy lại
3. **Package missing**: Chạy `!pip install package_name`
4. **Session timeout**: Re-run cells từ đầu

**Tips:**
- Save notebook vào Drive để backup
- Download results quan trọng về máy
- Sử dụng GPU nếu cần tốc độ cao

## 🎉 Thành công!

Sau khi hoàn thành, bạn sẽ có:
- 📊 Complete ML project analysis
- 📈 Model performance comparison  
- 🎯 Feature selection insights
- 💾 Downloadable results
- 📱 Shareable Colab notebook

**Thời gian ước tính: 5-10 phút**
**Không cần cài đặt gì trên máy!**

---

**🚀 Ready to start your ML journey on Colab!** 