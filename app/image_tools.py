import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


from skimage import io
from skimage import img_as_float, img_as_ubyte

class ImageTools:

    def __init__(self):
        sns.set_style('ticks')

    def get_image(self, img_bytes):
        return io.imread(img_bytes, as_gray=True, plugin='pil')

    def show_image(self, img):
        plt.imshow(img, cmap='gray')

    def show_all_images(self, *images, titles=None, cmap='gray'):
        # TODO: Refact
        images = [img_as_float(image) for image in images]

        if titles is None:
            titles = [''] * len(images)

        vmin = min(map(np.min, images))
        vmax = max(map(np.max, images))
        ncols = len(images)
        height = 6
        width = height * len(images)
        fig, axes = plt.subplots(nrows=1, ncols=ncols,
                                figsize=(width, height))
        
        for ax, img, label in zip(axes.ravel(), images, titles):
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_title(label)
