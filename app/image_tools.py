import matplotlib.pyplot as plt
import seaborn as sns

from skimage import io
from skimage import img_as_float, img_as_ubyte

from skimage.morphology import square, disk

from skimage.filters import gaussian as gaussian_filter
from skimage.filters.rank import mean as mean_filter, median as median_filter

from skimage.exposure import equalize_hist

from skimage.draw import circle


class ImageTools:

    def __init__(self, np):
        self.np = np
        sns.set_style('ticks')

    def get_image(self, img_bytes):
        return io.imread(img_bytes, as_gray=True, plugin='pil')

    def show_image(self, img):
        plt.imshow(img, cmap='gray')

    def show_all_images(self, images, titles=None):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.tight_layout()
        for ax, img, label in zip(axes.ravel(), images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(label)

    def get_selem(self, width):
        return square(width)

    def get_filtered_images(self, image, selem, sigma):
        filtered_images = dict()
        filtered_images['Média'] = self.filter_image('Média', image, selem, sigma)
        filtered_images['Mediana'] = self.filter_image('Mediana', image, selem, sigma)
        filtered_images['Gaussiano'] = self.filter_image('Gaussiano', image, selem, sigma)
        filtered_images['Equalização de Histograma'] = self.filter_image('Equalização de Histograma', image, selem, sigma)
        return filtered_images

    def filter_image(self, type_filter, image, selem, sigma):
        if type_filter == 'Média':
            return mean_filter(img_as_ubyte(image), selem)
        elif type_filter == 'Mediana':
            return median_filter(img_as_ubyte(image), selem)
        elif type_filter == 'Gaussiano':
            return gaussian_filter(image, sigma)
        elif type_filter == 'Equalização de Histograma':
            return equalize_hist(image)
        raise ValueError('Invalid type of filter!')

    def remove_flashes(self, image, L=6, k=1.5):
        print('Removendo reflexos especulares...')
    
        lim_min, lim_row_max, lim_col_max, = L+1, image.shape[0]-L-1, image.shape[1]-L-1

        # Para cada pixel (x, y)
        for r in range(lim_min, lim_row_max):
            for c in range(lim_min, lim_col_max):
                # Calcula o valor médio entre o pixel e sua 4-vizinhança
                v0 = self.__mean_value_neighborhood(image, pixel=(r, c))
                
                # Calcula o valor médio dos pixels L distantes
                vr = self.__mean_value_neighborhood(image, pixel=(r+L, c))
                vl = self.__mean_value_neighborhood(image, pixel=(r-L, c))
                vu = self.__mean_value_neighborhood(image, pixel=(r, c+L))
                vd = self.__mean_value_neighborhood(image, pixel=(r, c-L))

                # Verifica se pixel está próximo ao centro do artefato a ser removido
                change_pixel = (v0 > vr*k) and (v0 > vl*k) and (vu > vr*k) and (vd > vr*k)
                if change_pixel:
                    print('Artefato no pixel:', (r, c))
                    self.__change_value_neighborhood(image, (r, c), L)

    def __mean_value_neighborhood(self, image, pixel):
        r, c = pixel[0], pixel[1]
        return self.np.mean([image[r, c], image[r+1, c], image[r-1, c], image[r, c+1], image[r, c-1]])

    def __change_value_neighborhood(self, image, pixel, L):
        r, c = pixel[0], pixel[1]
        
        # Acha as coordenadas de um círculo de raio L centrado em (x, y)
        r_coords, c_coords = circle(r, c, L, image.shape)

        # Calcula o valor mínimo entre todos os pixel localizados a uma distância menor que L 
        new_value = image[r_coords, c_coords].min()

        # Substitui o pixel e seus vizinhos pelo novo valor
        image[r_coords, c_coords] = new_value
