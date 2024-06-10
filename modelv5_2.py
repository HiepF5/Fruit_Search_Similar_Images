import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Sequential
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

# Save extracted features and image paths to files
np.save('train_features.npy', train_features)  # Save extracted features to 'train_features.npy' file
np.save('train_image_paths.npy', train_image_paths)  # Save image paths to 'train_image_paths.npy' file

# Convert the features to a DataFrame and save as CSV
df_features = pd.DataFrame(train_features)
df_features.to_csv('train_features.csv', index=False)

# Save image paths to CSV
df_paths = pd.DataFrame(train_image_paths, columns=['image_path'])
df_paths.to_csv('train_image_paths.csv', index=False)
