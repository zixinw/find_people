import keras.optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.initializers import random_normal
from keras.utils import plot_model
from data import get_X_and_y, get10_30_X_and_y
from sklearn.utils import shuffle

model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer=random_normal(), input_shape=(30, 10, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer=random_normal()))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(16, (2, 2), activation='relu', kernel_initializer=random_normal()))
model.add(Conv2D(16, (1, 1), activation='relu', kernel_initializer=random_normal()))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=random_normal()))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax', kernel_initializer=random_normal()))

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# X, y = get_X_and_y('./train')
X, y, files = get10_30_X_and_y('./labeled_data/people', negativePath='./dummies/10_30')

# X, y = shuffle(X,y)
# print(y)


model.fit(X, y, batch_size=2, epochs=50, shuffle=True)
model.save('vgg2.h5')
# plot_model(model)
