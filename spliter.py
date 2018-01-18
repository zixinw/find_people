import os
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize


class Spliter(object):
    def __init__(self, sourcePath, destination):
        self.sourcePath = sourcePath
        self.destination = destination
        dir = destination
        if not os.path.exists(dir) or not os.path.isdir(dir):
            os.mkdir(dir)

    def slide_window(self, fileName, stride=50, window=(500, 500), width=1242, height=(700, 1600)):

        min_height = height[0]
        max_height = height[1]
        x_index = window[0]
        y_index = window[1] + min_height

        frames = []

        img = io.imread('{}/{}'.format(self.sourcePath, fileName))
        print('img.ndim ', img.ndim)
        if img.ndim > 2:
            img = rgba2rgb(img)
            img = rgb2gray(img)

        while x_index < width or y_index < max_height:

            if (x_index < width and y_index < max_height):
                # format (left, top, right, bottom)
                frames.append((x_index - window[0], y_index - window[1], x_index, y_index))

            if (x_index >= width):
                x_index = window[0]
                y_index += stride

            x_index += stride

        for index, frame in enumerate(frames):
            print(frame)
            dir = '{}/{}'.format(self.destination, fileName[:-4])

            snapshot = img[frame[1]:frame[3], frame[0]:frame[2]]
            if not os.path.exists(dir) or not os.path.isdir(dir):
                os.mkdir(dir)
            io.imsave('{}/{}_N.jpg'.format(dir, index), snapshot)
            # io.imsave('{}/{}_N.jpg'.format(dir, index), resize(snapshot, (32, 32)))

    # def digestImgs(self):
    #     files = os.listdir(self.sourcePath)
    #     for file in files:
    #         if file.endswith('.PNG'):
    #             self.slide_window(file)

    def batch_process(self, process, **kwargs):
        files = os.listdir(self.sourcePath)
        print(files)
        for file in files:
            if file.endswith('.jpg'):
                process(file, **kwargs)


spliter = Spliter(sourcePath='./train_crop', destination='./test', )

spliter.batch_process(process=spliter.slide_window, stride=5, window=(10, 30),
                      width=124, height=(0, 80))
# Spliter(sourcePath='./jumpgame_dataset', destination='./train').digestImgs()
