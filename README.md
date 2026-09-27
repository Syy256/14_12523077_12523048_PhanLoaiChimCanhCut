# Penguin Classification

Ứng dụng dự đoán loài chim cánh cụt từ dataset Palmer Archipelago. Giao diện có 6 trường đầu vào và tự động cập nhật kết quả của 4 model: Logistic Regression, KNN, SVM và Decision Tree. Mỗi kết quả hiển thị confidence, xác suất theo loài và Accuracy / Precision / Recall / F1 trên tập test.

## Chạy bằng Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/docs

Dataset được đọc trực tiếp từ `AI-models/data/archive.zip`, không cần giải nén thủ công.
