import os
from keras.preprocessing.image import img_to_array, load_img
import csv
from keras.models import Model
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Input
from keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy



# This part is prepartion part to load csv file and image folder
images_directory = r"C:\Users\bmnba\Desktop\hepsi2"
labels_directory = r"C:\Users\bmnba\Desktop\Kitap2.csv"

# Creating empty values that will needed soon
images_list = []
x_values_list = []
y_values_list = []
bool_values_list = []
x_train = []
y_train = []
bool_train = []
image_train = []
x_test = []
y_test = []
bool_test = []
image_test = []

# Reading csv files and giving assising them to variables
with open(labels_directory, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)  # Skip header if present
    for row in reader:
        x_values_list.append(float(row[0])/50)
        y_values_list.append(float(row[1])/50)
        bool_values_list.append(float(row[2]))

# Loading images and converting them to array, meanwhile creating test and train datasets
for index in range(1, min(101, len(x_values_list) + 1)):
    img_name = f"{index}.jpg"
    img_path = os.path.join(images_directory, img_name)

    # Load image and convert to array
    img = img_to_array(load_img(img_path, target_size=(256, 256)))
    img /= 255.0  # Normalize pixel values between 0 and 1

    if (index % 5) == 0:
        image_test.append(img)
        x_test.append(x_values_list[index - 1])
        y_test.append(y_values_list[index - 1])
        bool_test.append(bool_values_list[index - 1])
    else:
        image_train.append(img)
        x_train.append(x_values_list[index - 1])
        y_train.append(y_values_list[index - 1])
        bool_train.append(bool_values_list[index - 1])

# making all variables numpy array to be used in NN
x_train = np.array(x_train)
y_train = np.array(y_train)
bool_train = np.array(bool_train)
image_train = np.array(image_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
bool_test = np.array(bool_test)
image_test = np.array(image_test)

# Building model, inputs are x-y axis and image
image_input = Input(shape=(256, 256, 3), name='image_input')
x_input = Input(shape=(1,), name='x_input')
y_input = Input(shape=(1,), name='y_input')


# Processing image with convolutional layers
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flatten = Flatten()(pool2)

# Combining x-y axis with image
concatenated = concatenate([flatten, x_input, y_input], name='concatenated')

# Lastly dense layers for binary classification
dense1 = Dense(64, activation='relu')(concatenated)
output_bool = Dense(1, activation='sigmoid', name='output_bool')(dense1)

# Model creation
model_binary = Model(inputs=[image_input, x_input, y_input], outputs=output_bool)

# Compiling the model with Adam optimization algorithm
model_binary.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model with the dataset that I created before
model_binary.fit([image_train, x_train, y_train], bool_train, epochs=10, batch_size=32, validation_data=([image_test, x_test, y_test], bool_test))

# Testing the model with the test dataset
predictions = model_binary.predict([image_test, x_test, y_test])

# Since outcome is either 1 or 0, I created threshold for 0.5
threshold = 0.5
predicted_bool = (predictions > threshold)

# Calculating the accuracy and printing it
accuracy_metric = Accuracy()
accuracy_metric.update_state(bool_test, predicted_bool)
accuracy = accuracy_metric.result().numpy() * 100
print("Accuracy:", accuracy)