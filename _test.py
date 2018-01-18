import keras
import data
import numpy as np

X, y, files = data.get10_30_X_and_y(positivePath='./falsePositives', negativePath='dummy_10_30')
# X, y = data.get_X_and_y('./train')

print('X.shape is ', X.shape)

model = keras.models.load_model('vgg1.h5')
print(model.evaluate(X, y))

predictions = model.predict(X[:52])
print('falsePositive precision:')
print(np.mean(np.rint(predictions[:,0])))

predictions = model.predict(X[52:])
print('dummy precision:')
print(np.mean(np.rint(predictions[:,0])))


X, y, files = data.get10_30_X_and_y(positivePath='./class_people', negativePath='./class_box')

print('X.shape is ', X.shape)

model = keras.models.load_model('vgg1.h5')
print(model.evaluate(X, y))

predictions = model.predict(X[:52])
print('class people precision:')
print(np.mean(np.rint(predictions[:,1])))

predictions = model.predict(X[52:])
print('class box precision:')
print(np.mean(np.rint(predictions[:,0])))
