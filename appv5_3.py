import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define your custom CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Load paths of images in the 'Black2/train' directory
train_image_paths = []
for root, _, files in os.walk("Black2/train"):
    for file in files:
        if file.endswith((".jpg", ".png")):
            train_image_paths.append(os.path.join(root, file))

# Extract features of all images in the 'Black2/train' directory
train_features = []
for image_path in train_image_paths:
    img = image.load_img(image_path, target_size=(500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    features = model.predict(img)
    train_features.append(features.flatten())
print(train_features)
# Save extracted features and image paths to files
np.save('train_features.npy', train_features)  # Save extracted features to 'train_features.npy' file
np.save('train_image_paths.npy', train_image_paths)  # Save image paths to 'train_image_paths.npy' file

# Function to calculate cosine similarity
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

# Function to display similar images in a new window
def display_similar_images(similar_images, window_title):
    similar_window = tk.Toplevel()
    similar_window.title(window_title)

    for image_path, similarity in similar_images:
        similarity_scalar = similarity.item()
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

# Function to find and display similar images
def find_and_display_similar_images():
    input_image_path = selected_image_path.get()
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((500, 500))
    input_img_photo = ImageTk.PhotoImage(input_image)

    input_img_label.config(image=input_img_photo)
    input_img_label.image = input_img_photo

    img = image.load_img(input_image_path, target_size=(500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    input_feature = model.predict(img).flatten()

    similarities = []
    for train_feature, train_image_path in zip(train_features, train_image_paths):
        similarity = cosine_similarity(input_feature, train_feature)
        similarities.append((train_image_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarities[:3]

    display_similar_images(top_similar_images, "Top 3 Similar Images")

# Create main Tkinter application
last_directory = "/"
app = tk.Tk()
app.title("Similar Images Finder")

# Function to open file dialog and display selected image
def open_file_dialog():
    global last_directory
    file_path = filedialog.askopenfilename(initialdir=last_directory, title="Select Image",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        last_directory = os.path.dirname(file_path)
        display_image(file_path)

# Function to display selected image
def display_image(file_path):
    global img
    image = Image.open(file_path)
    image = image.resize((100, 100))
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img
    selected_image_path.set(file_path)

# Set up dimensions and position of the application window
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
content_width = 850
content_height = 850
x = (screen_width - content_width) // 2
y = (screen_height - content_height) // 2
app.geometry(f"{content_width}x{content_height}+{x}+{y}")

# Create GUI components
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
