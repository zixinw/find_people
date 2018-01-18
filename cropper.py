import os
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import rescale


##
# 用于原始截图，截取 800 - 1600 高度像素部门，并作0.1比例缩放
##

class Cropper(object):
    def __init__(self, sourcePath, destination):
        self.sourcePath = sourcePath
        self.destination = destination
        self.count = 0
        if not os.path.exists(self.destination) or not os.path.isdir(self.destination):
            os.mkdir(self.destination)

    def digestImg(self, fileName, height=(800, 1600)):

        min_height = height[0]
        max_height = height[1]

        img = rgba2rgb(io.imread('{}/{}'.format(self.sourcePath, fileName)))
        img = rgb2gray(img)

        snapshot = img[min_height:max_height]
        self.count += 1
        io.imsave('{}/{}.jpg'.format(self.destination, self.count), rescale(snapshot, 0.1))

    def digestImgs(self):
        files = os.listdir(self.sourcePath)
        for file in files:
            if file.endswith('.PNG'):
                self.digestImg(file)


Cropper(sourcePath='./screen_shots', destination='./train_crop').digestImgs()
