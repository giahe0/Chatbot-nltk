# Ted, The Deep-Learning Chatbot
## Về dự án này
Ted là một chatbot đa năng được tạo bằng Python3, có thể trò chuyện với bạn và giúp bạn thực hiện các công việc hàng ngày. Nó sử dụng NLP và Deep-Learning để phân tích tin nhắn của người dùng, phân loại nó thành danh mục rộng hơn và sau đó trả lời bằng tin nhắn phù hợp hoặc thông tin được yêu cầu.
## Project UI
Trang chủ:
![image](image.png)

Để nó chạy được trên hệ thống của bạn, thì làm theo các bước sau:
1. Sao chép kho lưu trữ này vào hệ thống của bạn. Trên Command Prompt, chạy lệnh sau:
```
git clone https://github.com/giahe0/Chatbot-nltk.git
```
2. Thay đổi thư mục của bạn thành Chatbot-nltk:
```
cd Chatbot-nltk
```
3. Đảm bảo rằng bạn có tất cả các thư viện cần thiết được liệt kê trong requirements.txt. Trong trường hợp thiếu bất kỳ thư viện nào, hãy cài đặt chúng bằng pip. Nhập lệnh này vào Command Prompt của bạn, thay thế 'Your-library-name' bằng tên thư viện được yêu cầu:
```
pip install Your-library-name 
```
4. Sau đó chạy các lệnh sau để chạy ứng dụng:
```
set FLASK_APP=chatbot.py
flask run
```
5. Nhập url được cung cấp sau khi chạy các lệnh trước đó vào trình duyệt web của bạn

Ted bây giờ đã sẵn sàng để trò chuyện!

##### Copyright (c) 2020 Karan-Malik