import os
import numpy as np
from skimage import io
import keras


def get_X_and_y(path):
    train = path
    image_dirs = list(map(lambda x: '{}/{}'.format(train, x), os.listdir(train)))
    image_dirs = list(filter(lambda x: os.path.isdir(x), image_dirs))

    images = ['{}/{}'.format(dir, file) for dir in image_dirs for file in os.listdir(dir)]
    images = list(filter(lambda x: x.endswith('.jpg'), images))

    X = np.asarray([io.imread(f) for f in images])
    X = X.reshape(X.shape + (1,))
    print(X.shape)

    categories = {'N': 0, "A": 1, 'C': 2}
    y = keras.utils.to_categorical(np.asarray(list(map(lambda x: categories[x[-5:-4]], images))))
    print(y.shape)
    return X, y

    # get_X_and_y('./train')


def get10_30_X_and_y(positivePath, negativePath):
    positiveFiles = os.listdir(positivePath)
    negativeFiles = os.listdir(negativePath)

    def filter_dotDS(files):
        return list(filter(lambda x: not x.startswith('.DS_'), files))

    positiveFiles = filter_dotDS(positiveFiles)
    negativeFiles = filter_dotDS(negativeFiles)

    print(positiveFiles)
    X = np.asarray([io.imread('{}/{}'.format(positivePath, f)) for f in positiveFiles] +
                   [io.imread('{}/{}'.format(negativePath, f)) for f in negativeFiles])

    X = X.reshape(X.shape + (1,))
    print(X.shape)
    y = np.concatenate((np.ones((len(positiveFiles)), dtype=int),  np.zeros((len(negativeFiles)), dtype=int)))
    y = keras.utils.to_categorical(y)
    print(y)

    # io.imshow(X[0].reshape((30,10)))
    # io.show()
    files = positiveFiles + negativeFiles
    return X, y, files


# get10_30_X_and_y('./class_people', negativePath='dummy_10_30')
