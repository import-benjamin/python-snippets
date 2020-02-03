from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import subprocess
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Flatten, Dropout, Dense

print(subprocess.run("git clone https://github.com/aryapei/In-shop-Clothes-From-Deepfashion.git", shell=True,  stdout=subprocess.PIPE).stdout.decode("utf-8"))

conv_base = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(14, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, decay=1e-6), metrics=['acc'])
model.summary()


train_generator = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)

train_generator = train_generator.flow_from_directory("./In-shop-Clothes-From-Deepfashion/Img/WOMEN/",
                                                      batch_size=45,
                                                      target_size=(224,224))

model.fit(train_generator,
          steps_per_epoch=45,
          epochs=32)

model.save('weights.h5')
