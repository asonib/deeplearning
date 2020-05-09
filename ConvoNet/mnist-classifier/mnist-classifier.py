from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

print('Shape of X_train', str(X_train.shape))
print('Number of sample in training dataset', str(len(X_train)))
print('number of labels in the test dataset', str(len(y_train)))

print('Number of sample in testing dataset', str(len(X_test)))
print('number of labels in the test dataset', str(len(y_test)))

print('Dimensions of X_test', str(X_test[0].shape))
print('Labels in y_test', str(y_test.shape))

#Dispaly some image using openCV
import cv2
import numpy as np
for i in range(0, 6):
    img = X_train[np.random.randint(0, len(X_train))]
    name = "Sample"+str(i)
    
    cv2.imshow(name, img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

import matplotlib.pyplot as plt

for i in range(0, 3):
    plt.subplot(331)
    plt.imshow(X_train[np.random.randint(0, len(X_train))])
    plt.show()


img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)

##OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('Number of classes', str(y_test.shape[0]))

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape = input_shape, activation = 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = SGD(0.01))

print(model.summary())

batch_size = 32
epochs = 10

history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1,
                   validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test, verbose = 0)
print('Test Loss', score[0])
print('Test Accuracy', score[1])


import matplotlib.pyplot as plt

history_dic = history.history

loss_values = history_dic['loss']
val_loss_values = history_dic['val_loss']
epochs = range(1, len(loss_values)+1)

line1 = plt.plot(epochs, val_loss_values, label='Valdation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

import matplotlib.pyplot as plt

history_dic = history.history

acc_values = history_dic['accuracy']
val_acc_values = history_dic['val_accuracy']
epochs = range(1, len(acc_values)+1)

line1 = plt.plot(epochs, val_acc_values, label='Valdation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

model.save('./mnist-2.h5')
from keras.models import load_model
classifier = load_model('./mnist-2.h5')

#sudo apt-get install graphviz
from keras.utils.vis_utils import plot_model

plot_model(model,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB')
