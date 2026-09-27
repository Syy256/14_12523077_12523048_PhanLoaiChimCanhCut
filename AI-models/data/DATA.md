# DATA.md

## Nguồn dữ liệu

* Tên dataset: Palmer Archipelago (Antarctica) Penguin Data
* Link: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
* Tác giả/đơn vị cung cấp: Parul Pandey (dataset trên Kaggle); dữ liệu gốc được thu thập và cung cấp bởi Dr. Kristen Gorman và Palmer Station Antarctica LTER
* Giấy phép (license): CC0
* Ngày tải: 24/09/2026

## Mô tả bài toán

* Loại bài toán: Phân loại đa lớp (Multiclass Classification)
* Cột mục tiêu (target): `species`, ý nghĩa: Phân loại chim cánh cụt theo loài dựa trên các đặc điểm hình thái và thông tin quan sát.
* Số lớp / giá trị của target: 3 lớp `{Adelie, Chinstrap, Gentoo}`

  * `Adelie`: Chim cánh cụt Adélie
  * `Chinstrap`: Chim cánh cụt Chinstrap
  * `Gentoo`: Chim cánh cụt Gentoo

## Mô tả các cột dữ liệu

| Cột                    | Kiểu dữ liệu | Ý nghĩa                                      | Ghi chú                                      |
| ---------------------- | ------------ | --------------------------------------------- | -------------------------------------------- |
| `species`              | object       | Loài chim cánh cụt                            | Target; `Adelie`, `Chinstrap`, `Gentoo`      |
| `island`               | object       | Đảo nơi quan sát chim cánh cụt                | `Biscoe`, `Dream`, `Torgersen`               |
| `culmen_length_mm`     | float        | Chiều dài mỏ                                  | Đơn vị mm                                    |
| `culmen_depth_mm`      | float        | Độ sâu mỏ                                     | Đơn vị mm                                    |
| `flipper_length_mm`    | int          | Chiều dài cánh chèo                           | Đơn vị mm                                    |
| `body_mass_g`          | int          | Khối lượng cơ thể                             | Đơn vị gram                                  |
| `sex`                  | object       | Giới tính của chim cánh cụt                   | `MALE`, `FEMALE`; có một số giá trị thiếu     |

## Số lượng

* Số dòng: 344 dòng trong `penguins_size.csv`
* Số cột: 7 cột
* Tỉ lệ thiếu dữ liệu: Có dữ liệu thiếu ở `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g` và `sex`.
* Các cột `species` và `island` không có dữ liệu thiếu trong phiên bản `penguins_size.csv`.
* Phân bố nhãn (classification): 3 lớp `Adelie`, `Chinstrap`, `Gentoo`.

## Các file dữ liệu

### penguins_size.csv

* Số dòng: 344
* Số cột: 7
* Bao gồm 6 feature và 1 target `species`.
* Các feature gồm thông tin về đảo, kích thước mỏ, chiều dài cánh chèo, khối lượng cơ thể và giới tính.
* Được sử dụng để phân tích dữ liệu, trực quan hóa và huấn luyện các mô hình Machine Learning.
* Dataset này là phiên bản đơn giản hóa của dữ liệu chim cánh cụt Palmer và là file phù hợp để sử dụng cho các bài toán nhập môn về phân tích dữ liệu và Machine Learning.

### penguins_lter.csv

* Là file dữ liệu khác được cung cấp trong dataset Kaggle.
* Chứa dữ liệu gốc/chi tiết hơn về các quan sát chim cánh cụt tại Palmer Archipelago.
* Có nhiều thông tin hơn so với `penguins_size.csv`; cấu trúc và số lượng cột không giống file dữ liệu đơn giản hóa.
* Khi xây dựng bài toán Machine Learning theo cấu trúc của DATA.md này, sử dụng `penguins_size.csv` làm dataset chính.

## Mục tiêu Machine Learning

Xây dựng các mô hình Machine Learning để dự đoán `species` của một con chim cánh cụt dựa trên các đặc điểm trong dataset.

Có thể sử dụng các mô hình:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)

Các mô hình sẽ được đánh giá bằng:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Đặc biệt, mỗi mô hình cần có Confusion Matrix để đánh giá khả năng phân loại 3 loài chim cánh cụt: `Adelie`, `Chinstrap` và `Gentoo`.
