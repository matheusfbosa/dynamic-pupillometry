import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from skimage import io
from skimage import img_as_float, img_as_ubyte

from skimage.morphology import square, disk
from skimage.filters import gaussian as gaussian_filter, threshold_otsu
from skimage.filters import sobel, sobel_v, sobel_h
from skimage.filters import scharr, scharr_v, scharr_h
from skimage.filters import prewitt, prewitt_v, prewitt_h
from skimage.filters.rank import mean as mean_filter, median as median_filter
from skimage.exposure import equalize_hist
from skimage.draw import disk as draw_disk, circle_perimeter
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks



class Pupillometry:


    def __init__(self):
        self.image_in = None
        self.image_filtered = None
        self.image_pre_processed = None
        self.region_pupil = None
        self.image_pupil = None
        self.region_iris = None
        self.image_iris = None
        self.image_out = None
        self.type_of_filters = ['Média', 'Mediana', 'Gaussiano', 'Equalização de Histograma']
        sns.set_style('ticks')


    def get_np_array(self, array):
        return np.array(array)


    def get_image(self, image_bytes):
        self.image_in = io.imread(image_bytes, as_gray=True, plugin='pil')


    def show_image(self, image):
        plt.imshow(image, cmap='gray')


    def show_all_images(self, images, nrows, ncols, titles=None):
        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        for ax, img, label in zip(axes.ravel(), images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(label)


    def get_selem(self, width):
        return square(width)


    def get_filtered_images(self, selem, sigma, image):
        images_filtered = dict()
        for type_of_filter in self.type_of_filters:
            images_filtered[type_of_filter] = self.filter_image(image, selem, sigma, type_of_filter)
        return images_filtered


    def filter_image(self, image, selem, sigma, type_of_filter):
        # Filtros de suavização
        if type_of_filter == 'Média':
            return mean_filter(img_as_ubyte(image), selem)
        elif type_of_filter == 'Mediana':
            return median_filter(img_as_ubyte(image), selem)
        elif type_of_filter == 'Gaussiano':
            return gaussian_filter(image, sigma)
        elif type_of_filter == 'Equalização de Histograma':
            return equalize_hist(image)

        # Filtros de realce
        elif type_of_filter == 'Prewitt Horizontal':
            return prewitt_h(image)
        elif type_of_filter == 'Prewitt Vertical':
            return prewitt_v(image)
        elif type_of_filter == 'Prewitt':
            return prewitt(image)
        elif type_of_filter == 'Sobel Horizontal':
            return sobel_h(image)
        elif type_of_filter == 'Sobel Vertical':
            return sobel_v(image)
        elif type_of_filter == 'Sobel':
            return sobel(image)
        elif type_of_filter == 'Scharr Horizontal':
            return scharr_h(image)
        elif type_of_filter == 'Scharr Vertical':
            return scharr_v(image)
        elif type_of_filter == 'Scharr':
            return scharr(image)
        
        raise ValueError(f'Invalid type of filter! {type_of_filter} is not valid.')


    def remove_flashes(self, L, k, image):
        artifacts = []
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
                    artifacts.append((r, c))
                    self.__change_value_neighborhood(image, (r, c), L)
        
        return artifacts


    def __mean_value_neighborhood(self, image, pixel):
        r, c = pixel[0], pixel[1]
        return np.mean([image[r, c], image[r+1, c], image[r-1, c], image[r, c+1], image[r, c-1]])


    def __change_value_neighborhood(self, image, pixel, L):
        r, c = pixel[0], pixel[1]
        
        # Acha as coordenadas de um círculo de raio L centrado em (x, y)
        r_coords, c_coords = draw_disk((r, c), L, shape=image.shape)

        # Calcula o valor mínimo entre todos os pixel localizados a uma distância menor que L 
        new_value = image[r_coords, c_coords].min()

        # Substitui o pixel e seus vizinhos pelo novo valor
        image[r_coords, c_coords] = new_value


    def get_otsu_threshold(self, image):
        return threshold_otsu(image)


    def get_edges_canny(self, sigma, low_threshold, high_threshold, image):
        return canny(image, sigma, low_threshold, high_threshold)


    def pupil_segmentation_hough(self, image):
        # Fornece uma gama de raios plausíveis
        hough_radii = np.arange(40, 80, 2)

        hough_res = hough_circle(image, hough_radii)
        _, cols, rows, rads = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

        # Seleciona a região com maior raio encontrado
        region = max(zip(rows, cols, rads), key=lambda item: item[2])
        return region


    def draw_circle_perimeter(self, image, region):
        r_coords_pupil, c_coords_pupil = circle_perimeter(region[0], region[1], region[2], shape=image.shape)
        image[r_coords_pupil, c_coords_pupil] = 1
        return image


    def iris_segmentation_horizontal_gradient(self, image, a=20, b=120):
        # offset_col = col_pupil + radius_pupil + a
        offset_col = self.region_pupil[1] + self.region_pupil[2] + a
        gradient_col = np.gradient(image[self.region_pupil[0], offset_col:(offset_col + b)])

        col_edge = np.argmax(gradient_col) + offset_col
        rad = col_edge - self.region_pupil[0]

        # region = row, column, radius
        region = (self.region_pupil[0], self.region_pupil[1], rad)
        return region
