import selectivesearch as ss
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpathces
import keras
import numpy as np

model = keras.models.load_model('vgg1.h5')

img = io.imread('./jumpgame_dataset/IMG_2527.PNG')
img = rgba2rgb(img)
img = rescale(img, 0.2)
img_lbl, regions = ss.selective_search(img)

candidates = set()
for r in regions:
    if r['rect'] in candidates:
        continue

    # if r['size'] > 5000 or r['size'] < 1000:
    #     continue

    x, y, w, h = r['rect']

    if w is 0 or h is 0:
        continue

    distortRate = 7
    if w / h > distortRate or h / w > distortRate:
        continue

    candidates.add(r['rect'])

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6, 6))
ax = axes[0]
ax.imshow(img)
print(img.shape)

for x, y, w, h in candidates:
    sample = img[y:y + h, x:x + w]
    prediction = model.predict(resize(rgb2gray(sample), (30, 10)).reshape((1, 30, 10, 1)))
    print(x, y, w, h)
    print(prediction)
    # if prediction[0][1] > 0.9:
    rect = mpathces.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    axes[1].imshow(sample)

plt.show()
