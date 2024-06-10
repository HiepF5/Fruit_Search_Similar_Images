import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Tải mô hình MobileNetV2 đã được huấn luyện sẵn để trích xuất đặc trưng
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(500, 500, 3))

# Tải đường dẫn của các ảnh trong thư mục 'Black2/train'
train_image_paths = []
for root, _, files in os.walk("Black2/train"):
    for file in files:
        if file.endswith((".jpg", ".png")):
            train_image_paths.append(os.path.join(root, file))

# Trích xuất đặc trưng của tất cả các ảnh trong thư mục 'Black2/train'
train_features = []
for image_path in train_image_paths:
    img = image.load_img(image_path, target_size=(500, 500))  # Đọc ảnh và thay đổi kích thước ảnh
    img = image.img_to_array(img)  # Chuyển đổi ảnh thành mảng numpy
    img = np.expand_dims(img, axis=0)  # Thêm chiều để phù hợp với đầu vào của model
    img = preprocess_input(img)  # Tiền xử lý ảnh theo yêu cầu của MobileNetV2
    features = feature_extractor.predict(img)  # Trích xuất đặc trưng từ ảnh
    train_features.append(features.flatten())  # Thêm đặc trưng đã trích xuất vào danh sách

# Lưu các đặc trưng đã trích xuất vào tệp
np.save('train_features.npy', train_features)  # Lưu đặc trưng vào tệp 'train_features.npy'
np.save('train_image_paths.npy', train_image_paths)  # Lưu đường dẫn ảnh vào tệp 'train_image_paths.npy'
