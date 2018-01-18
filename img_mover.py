import keras
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

model = keras.models.load_model('vgg1.h5')

# predictions = model.predict(X[:52])
# print('people precision:')
# print(np.mean(np.rint(predictions[:,1])))
#
# predictions = model.predict(X[52:])
# print('dummy precision:')
# print(np.mean(np.rint(predictions[:,0])))

fig, axes = plt.subplots(ncols=10, nrows=5)


def getX(path):
    files = os.listdir(path)

    def filter_dotDS(files):
        return list(filter(lambda x: not x.startswith('.DS_'), files))

    files = filter_dotDS(files)

    X = np.asarray([io.imread('{}/{}'.format(path, f)) for f in files])

    X = X.reshape(X.shape + (1,))
    print(X.shape)
    return X, files


root = './screen_shots_cropped_frames_all'
axe_x = 0
axe_y = 0
for dir in os.listdir(root):
    path = '{}/{}'.format(root, dir)
    if os.path.isdir(path):
        X, files = getX(path)
        predictions = model.predict(X)
        predictions = predictions[:, 1] == max(predictions[:, 1])

        for index, p in enumerate(predictions):
            if p:
                # print(index, p, path, files[index])
                img_path = './{}/{}/{}'.format(root, dir, files[index])
                print(img_path, axe_x, axe_y)
                # os.rename('{}/{}'.format(path, files[index]), './class_people/vgg2_{}.jpg'.format(index))
                axes[axe_x][axe_y].imshow(io.imread(img_path))
                axe_y += 1
                if (axe_y == 10):
                    axe_y = 0
                    axe_x += 1

plt.show()
