import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Tải mô hình MobileNetV2 đã được huấn luyện sẵn để trích xuất đặc trưng
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, input_shape=(500, 500, 3))

# Tải các đặc trưng và đường dẫn ảnh đã được trích xuất
train_features = np.load('train_features.npy')
train_image_paths = np.load('train_image_paths.npy')

# Hàm tính toán cosine similarity
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

# Hàm hiển thị các ảnh tương tự trong một cửa sổ mới
def display_similar_images(similar_images, window_title):
    # Tạo một cửa sổ mới để hiển thị các ảnh tương tự
    similar_window = tk.Toplevel()
    similar_window.title(window_title)

    for image_path, similarity in similar_images:
        similarity_scalar = similarity.item()  # Chuyển numpy array thành số float
        image_frame = tk.Frame(similar_window)
        image_frame.pack(pady=5)

        label = tk.Label(image_frame, text=f"Similarity: {similarity_scalar:.2f}")
        label.pack()

        img = Image.open(image_path)
        img = img.resize((100, 100))
        img_photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(image_frame, image=img_photo)
        img_label.image = img_photo
        img_label.pack(side=tk.LEFT, padx=5)

# Hàm tìm và hiển thị các ảnh tương tự
def find_and_display_similar_images():
    # Tải ảnh đã chọn
    input_image_path = selected_image_path.get()
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((500, 500))
    input_img_photo = ImageTk.PhotoImage(input_image)

    # Hiển thị ảnh đã chọn trong cửa sổ chính
    input_img_label.config(image=input_img_photo)
    input_img_label.image = input_img_photo

    # Trích xuất đặc trưng của ảnh đã chọn
    img = image.load_img(input_image_path, target_size=(500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    input_feature = feature_extractor.predict(img).flatten()

    # Tính toán độ tương tự cosine với tất cả các ảnh trong tập huấn luyện
    similarities = []
    for train_feature, train_image_path in zip(train_features, train_image_paths):
        similarity = cosine_similarity(input_feature, train_feature)
        similarities.append((train_image_path, similarity))

    # Sắp xếp theo độ tương tự và chọn ra 3 ảnh tương tự nhất
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarities[:3]

    # Hiển thị các ảnh tương tự nhất trong một cửa sổ mới
    display_similar_images(top_similar_images, "Top 3 Similar Images")

# Tạo ứng dụng chính của Tkinter
last_directory = "/"
app = tk.Tk()
app.title("Similar Images Finder")

# Hàm mở hộp thoại chọn tệp và hiển thị ảnh đã chọn
def open_file_dialog():
    global last_directory
    file_path = filedialog.askopenfilename(initialdir=last_directory, title="Select Image",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        last_directory = os.path.dirname(file_path)
        display_image(file_path)

# Hàm hiển thị ảnh đã chọn
def display_image(file_path):
    global img
    image = Image.open(file_path)
    image = image.resize((100, 100))
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img
    selected_image_path.set(file_path)

# Thiết lập kích thước và vị trí của cửa sổ ứng dụng
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
content_width = 850
content_height = 850
x = (screen_width - content_width) // 2
y = (screen_height - content_height) // 2
app.geometry(f"{content_width}x{content_height}+{x}+{y}")

# Tạo các thành phần giao diện
title_label = tk.Label(app, text="Similar Images Finder", font=("Helvetica", 20, "bold"))
title_label.pack(side=tk.TOP, pady=15)
image_label = tk.Label(app)
image_label.pack()

selected_image_path = tk.StringVar()

input_img_label = tk.Label(app)
input_img_label.pack()

result_label = tk.Label(app, text="", font=("Helvetica", 12))
result_label.pack(pady=15)

button_frame = tk.Frame(app)
button_frame.pack(side=tk.BOTTOM, pady=15)

open_button = tk.Button(button_frame, text="Open Image", command=open_file_dialog)
open_button.pack(side=tk.LEFT, padx=5)

find_similar_button = tk.Button(button_frame, text="Find Similar Images", command=find_and_display_similar_images)
find_similar_button.pack(side=tk.LEFT, padx=5)

app.mainloop()
