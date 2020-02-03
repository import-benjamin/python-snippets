from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import subprocess
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Flatten, Dropout, Dense

!git clone https://github.com/aryapei/In-shop-Clothes-From-Deepfashion.git
!rsync -a ./In-shop-Clothes-From-Deepfashion/Img/MEN/ ./In-shop-Clothes-From-Deepfashion/Img/WOMEN/

conv_base = InceptionV3(input_shape=(299, 299, 3), weights="imagenet", include_top=False)
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(17, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, decay=1e-6), metrics=['acc'])
model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

batch_size = 100

print("Generating train dataset")
train_generator = train_datagen.flow_from_directory("./In-shop-Clothes-From-Deepfashion/Img/WOMEN/",
                                                    batch_size=batch_size,
                                                    target_size=(299, 299),
                                                    class_mode="categorical",
                                                    subset="training")
print("Generating validation dataset")
validation_generator = train_datagen.flow_from_directory("./In-shop-Clothes-From-Deepfashion/Img/WOMEN/",
                                                         batch_size=batch_size,
                                                         target_size=(299, 299),
                                                         class_mode="categorical",
                                                         subset="validation")

model.fit(train_generator,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=50)

model.save('plankton_mind.h5')
