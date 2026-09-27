# 🐧 Penguin Classification

Hệ thống phân loại loài chim cánh cụt sử dụng Machine Learning trên bộ dữ liệu **Palmer Penguins**.

## Giới thiệu

Dự án được xây dựng trong môn **Học máy cơ bản**, bao gồm đầy đủ quy trình:

* Khám phá dữ liệu (EDA)
* Tiền xử lý dữ liệu
* Huấn luyện nhiều mô hình Machine Learning
* Đánh giá và lựa chọn mô hình tốt nhất
* Đóng gói mô hình
* Triển khai AI Service
* Xây dựng Web Application dự đoán
* Docker hóa toàn bộ hệ thống

---

# Dataset

* Dataset: Palmer Penguins
* File: `AI-models/data/archive.zip`

Thông tin dataset và giấy phép sử dụng được mô tả trong:

```text
AI-models/data/DATA.md
```

---

# Cấu trúc project

```text
.
├── AI-models
│   ├── Colab
│   │   ├── 01_eda.ipynb
│   │   ├── 02_preprocess.ipynb
│   │   ├── 03_train.ipynb
│   │   └── 04_evaluate.ipynb
│   │
│   ├── data
│   ├── models
│   ├── service
│   └── requirements.txt
│
├── App
│   ├── Backend
│   └── Frontend
│
├── docs
│   └── figure
│
├── docker-compose.yml
├── README.md
└── .env.example
```

---

# Machine Learning Pipeline

Quy trình thực hiện:

1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Missing Value Handling
4. Feature Encoding
5. Feature Scaling
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Deployment

---

# Các mô hình được huấn luyện

* Logistic Regression- Mai
* K-Nearest Neighbors (KNN)- Mai
* Support Vector Machine (SVM)- Sỹ
* Gradient Boosting- Sỹ

Sau quá trình đánh giá, **Logistic Regression** được lựa chọn để triển khai.

---

Các service:

| Service     | URL                   |
| ----------- | --------------------- |
| Frontend    | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| AI Service  | http://localhost:8001 |
