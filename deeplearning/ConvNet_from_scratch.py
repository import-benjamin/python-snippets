import cv2
import numpy as np
import gc
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# To see our directory
import os
import random
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import TensorBoard

mpl.rcParams["toolbar"] = "None"

train_dir = "./data/train"
test_dir = "./data/test"

train_dogs = [
    "./data/train/{}".format(i) for i in os.listdir(train_dir) if "dog" in i
]  # get dog images
train_cats = [
    "./data/train/{}".format(i) for i in os.listdir(train_dir) if "cat" in i
]  # get cat images
test_imgs = [
    "./data/test/{}".format(i) for i in os.listdir(test_dir)
]  # get test images

train_imgs = (
    train_dogs[:2000] + train_cats[:2000]
)  # slice the dataset and use 2000 in each class
random.shuffle(train_imgs)  # shuffle it randomly

# to view an image from train_imgs:
# import matplotlib.image as mpimg
# for ima in train_imgs[0:3]:
#   img = mpimg.imread(ima)
#   imgplot = plt.imshow(img)
#   plt.show()

# Lets declare our image dimensions
# we are using coloured images.
nrows = 150
ncolumns = 150

channels = 3  # change to 1 if you want to use grayscale image

# A function to read and process the images to an acceptable format for our model


def read_and_process_image(list_of_images):
    """Returns two arrays: x is an array of resized imagesy is an array of labels"""
    x = []  # images
    y = []  # labels
    for image in list_of_images:
        im = cv2.imread(image, cv2.IMREAD_COLOR)
        im2 = cv2.resize(im, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
        x.append(im2)
        #  x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        # Read the image# get the labels
        if "dog" in image:
            y.append(1)
        elif "cat" in image:
            y.append(0)
    return x, y


# get the train and label data
x, y = read_and_process_image(train_imgs)

# Lets view some of the pics
# plt.figure(figsize=(20,10))
# columns = 5
# for i in range(columns):
#     plt.subplot(5 / columns + 1, columns, i + 1)
#     imgplot = plt.imshow(x[i])
#     plt.show()

# convert list to numpy array
x = np.array(x)
y = np.array(y)


print("Shape of train images is:", x.shape)
print("Shape of labels is:", y.shape)


# Lets split the data into train and test set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)

print("Shape of train images is:", x_train.shape)
print("Shape of validation images is:", x_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

# clear memory
del x
del y
gc.collect()

# get the length of the train and validation data
ntrain = len(x_train)
nval = len(x_val)

# We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32

model = models.Sequential()  # (1)
model.add(
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3))
)  # (2)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))  # (3)
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())  # (4)
model.add(layers.Dropout(0.5))  # Dropout for regularization# (5)
model.add(layers.Dense(512, activation="relu"))
model.add(
    layers.Dense(1, activation="sigmoid")
)  # (6)#Sigmoid function at the end because we have just two classes

model.summary()

model.compile(
    loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"]
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Scale the image between 0 and 1
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)  # We do not augment validation data. we only perform rescale

# Create the image generators
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

tensorboard = TensorBoard(log_dir="logs", write_graph=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=ntrain // batch_size,
    epochs=32,
    validation_data=val_generator,
    validation_steps=nval // batch_size,
    callbacks=[tensorboard],
)

model.save_weights("model_wieghts.h5")
model.save("model_keras.h5")

# lets plot the train and val curve
# get the details form the history object
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, "b", label="Training accuracy")
plt.plot(epochs, val_acc, "r", label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, "b", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()

x_test, y_test = read_and_process_image(
    test_imgs[0:10]
)  # Y_test in this case will be empty.
x = np.array(x_test)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

i = 0
text_labels = []
plt.figure(figsize=(30, 20))
columns = 5
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append("dog")
    else:
        text_labels.append("cat")
    plt.subplot(5 / columns + 1, columns, i + 1)  # columns ??
    plt.title("This is a " + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()
