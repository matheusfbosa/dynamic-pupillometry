import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from skimage import io
from skimage import img_as_float, img_as_ubyte

from skimage.color import label2rgb
from skimage.draw import disk as draw_disk, circle_perimeter
from skimage.exposure import equalize_hist
from skimage.feature import canny
from skimage.filters import gaussian as gaussian_filter, threshold_otsu
from skimage.filters import sobel, sobel_v, sobel_h
from skimage.filters import scharr, scharr_v, scharr_h
from skimage.filters.rank import mean as mean_filter, median as median_filter
from skimage.measure import regionprops, label as measure_label
from skimage.morphology import square, disk
from skimage.morphology import erosion, dilation, opening, closing
from skimage.transform import hough_circle, hough_circle_peaks



class Pupillometry:


    def __init__(self):
        self.image_in = None
        self.image_filtered = None
        self.image_pre_processed = None
        self.image_pupil = None
        self.image_iris = None
        self.image_out = None
        self.images_to_plot_blur = None
        self.images_to_plot_morph = None
        self.images_to_plot_highlight = None
        self.region_pupil = None
        self.region_iris = None
        self.artifacts_found = []
        self.blur_filters = ['Média', 'Mediana', 'Gaussiano', 'Equalização de histograma']
        self.highlight_filters = ['Sobel horizontal', 'Sobel vertical', 'Sobel', 'Scharr horizontal', 'Scharr vertical', 'Scharr']
        self.morphological_operators = ['Erosão', 'Dilatação', 'Abertura', 'Fechamento']
        sns.set_style('ticks')


    def reset(self):
        self.image_filtered = None
        self.image_pre_processed = None
        self.image_pupil = None
        self.image_iris = None
        self.image_out = None
        self.images_to_plot_blur = None
        self.images_to_plot_morph = None
        self.images_to_plot_highlight = None
        self.region_pupil = None
        self.region_iris = None
        self.artifacts_found = []


    def get_np_array(self, array):
        return np.array(array)


    def get_image(self, image_bytes):
        self.image_in = io.imread(image_bytes, as_gray=True, plugin='pil')


    def get_image_as_float(self, image):
        return img_as_float(image)


    def label_to_rgb(self, image_labels, image):
        return label2rgb(image_labels, image=image, bg_label=0)


    def show_image(self, image, cmap='gray'):
        plt.imshow(image, cmap)


    def show_image_and_histogram(self, image, image_bin, n_bins):
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.hist(image.ravel(), bins=n_bins)
        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(image_bin, cmap='gray')


    def show_all_images(self, images, nrows, ncols, titles=None, cmap='gray'):
        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        for ax, img, label in zip(axes.ravel(), images, titles):
            ax.imshow(img, cmap)
            ax.set_title(label)


    def get_histogram(self, image, n_bins):
        return np.histogram(image.ravel(), n_bins)


    def get_selem(self, width):
        return square(width)


    def get_blur_filtered_images(self, selem, sigma, image):
        images_filtered = dict()
        for type_of_filter in self.blur_filters:
            images_filtered[type_of_filter] = self.transform_image(image, selem, sigma, type_of_filter)
        return images_filtered

    
    def get_images_morphologically_transformed(self, selem, image):
        images_morph = dict()
        for type_of_operation in self.morphological_operators:
            images_morph[type_of_operation] = self.transform_image(image, selem, None, type_of_operation)
        return images_morph

    
    def get_highlight_filtered_images(self, selem, image):
        images_filtered = dict()
        for type_of_filter in self.highlight_filters:
            images_filtered[type_of_filter] = self.transform_image(image, selem, None, type_of_filter)
        return images_filtered


    def transform_image(self, image, selem, sigma, type_of_function):
        # Filtros de suavização
        if type_of_function == 'Média':
            return mean_filter(img_as_ubyte(image), selem)
        elif type_of_function == 'Mediana':
            return median_filter(img_as_ubyte(image), selem)
        elif type_of_function == 'Gaussiano':
            return gaussian_filter(image, sigma)
        elif type_of_function == 'Equalização de histograma':
            return equalize_hist(image)

        # Filtros de realce
        elif type_of_function == 'Sobel horizontal':
            return sobel_h(image)
        elif type_of_function == 'Sobel vertical':
            return sobel_v(image)
        elif type_of_function == 'Sobel':
            return sobel(image)
        elif type_of_function == 'Scharr horizontal':
            return scharr_h(image)
        elif type_of_function == 'Scharr vertical':
            return scharr_v(image)
        elif type_of_function == 'Scharr':
            return scharr(image)
        
        # Operadores morfológicos
        elif type_of_function == 'Erosão':
            return erosion(image, selem)
        elif type_of_function == 'Dilatação':
            return dilation(image, selem)
        elif type_of_function == 'Abertura':
            return opening(image, selem)
        elif type_of_function == 'Fechamento':
            return closing(image, selem)

        raise ValueError(f'Invalid type of function! {type_of_function} is not valid.')


    def handle_flashes(self, L, k, image):
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
        
        return image, artifacts


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


    def labelizer_image(self, image):
        return measure_label(image, background=0, return_num=True)


    def get_regionprops(self, image_labels):
        return regionprops(image_labels)


    def calculate_radius(self, area):
        return np.sqrt(area / np.pi)


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
