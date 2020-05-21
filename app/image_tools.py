import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.draw import circle
from skimage.morphology import square 


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

    def get_selem(width):
        return square(width)

    def generate_histograms():
        pass

    def __change_value_neighborhood(img, pixel, L):
        r, c = pixel[0], pixel[1]
        
        # Acha as coordenadas de um círculo de raio L centrado em (x, y)
        r_coords, c_coords = circle(r, c, L, img.shape)

        # Calcula o valor mínimo entre todos os pixel localizados a uma distância menor que L 
        new_value = img[r_coords, c_coords].min()

        # Substitui o pixel e seus vizinhos pelo novo valor
        img[r_coords, c_coords] = new_value

    def __mean_value_neighborhood(img, pixel):
        r, c = pixel[0], pixel[1]
        return np.mean([img[r, c], img[r+1, c], img[r-1, c], img[r, c+1], img[r, c-1]])

    def remove_flashes(img, L=6, k=1.5):
        print('Removendo reflexos especulares...')
    
        lim_min, lim_row_max, lim_col_max, = L+1, img.shape[0]-L-1, img.shape[1]-L-1

        # Para cada pixel (x, y)
        for r in range(lim_min, lim_row_max):
            for c in range(lim_min, lim_col_max):
                # Calcula o valor médio entre o pixel e sua 4-vizinhança
                v0 = __mean_value_neighborhood(img, pixel=(r, c))
                
                # Calcula o valor médio dos pixels L distantes
                vr = __mean_value_neighborhood(img, pixel=(r+L, c))
                vl = __mean_value_neighborhood(img, pixel=(r-L, c))
                vu = __mean_value_neighborhood(img, pixel=(r, c+L))
                vd = __mean_value_neighborhood(img, pixel=(r, c-L))

                # Verifica se pixel está próximo ao centro do artefato a ser removido
                change_pixel = (v0 > vr*k) and (v0 > vl*k) and (vu > vr*k) and (vd > vr*k)
                if change_pixel:
                    print('Artefato no pixel:', (r, c))
                    __change_value_neighborhood(img, (r, c), L)
        
        return img
