# Sentiment Analysis

> Một dự án demo giao diện đơn giản chạy mô hình học máy phân tích cảm xúc văn bản tiếng Việt.  
> **Môn học**: Công cụ lập trình trí tuệ nhân tạo  
> **Sinh viên thực hiện**: Đỗ Tiến Đạt - A47331

---

## 🚀 Giới thiệu

Dự án xây dựng một hệ thống trí tuệ nhân tạo dự đoán thông tin đầu ra dựa trên dữ liệu đầu vào từ người dùng. Hệ thống sử dụng mô hình học máy được huấn luyện sẵn và tích hợp vào giao diện web đơn giản thông qua Flask.

- Giao diện được chạy tại: `http://127.0.0.1:5000/`
- Mục tiêu: Triển khai thử nghiệm mô hình học máy có thể xử lý văn bản hoặc ảnh (tùy theo ứng dụng cụ thể)

---

## 🛠️ Tính năng nổi bật

- ✅ Giao diện người dùng thân thiện bằng HTML với Flask  
- ✅ Nhúng mô hình học máy huấn luyện sẵn  
- ✅ Dự đoán và phản hồi kết quả tức thời  
- ✅ Dễ dàng mở rộng và triển khai thực tế

---

## Cấu trúc thư mục

📦project-root
├── docx                       # File báo cáo 
├── pptx                       
├── preprocessing.py           # Tiền xử lý dữ liệu
├── train.py                   # Huấn luyện mô hình
├── app.py
├── svc_model.pkl/             # Mô hình đã huấn luyện
├── templates/
│   └── index.html             # Giao diện chính
└── README.md


