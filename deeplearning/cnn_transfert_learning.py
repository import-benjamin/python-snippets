from keras.applications import InceptionResNetV2
from keras import layers
from keras import models

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))

conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes

model.summary()

print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
