import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import imagehash
from PIL import Image

from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import label2rgb
from skimage.draw import disk as draw_disk, circle_perimeter
from skimage.exposure import equalize_hist
from skimage.feature import canny
from skimage.filters import gaussian as gaussian_filter
from skimage.filters import sobel_v, sobel
from skimage.filters import scharr_v, scharr
from skimage.filters import prewitt_v, prewitt
from skimage.filters.rank import mean as mean_filter, median as median_filter
from skimage.measure import regionprops, label as measure_label
from skimage.morphology import square, disk
from skimage.morphology import erosion, dilation, opening, closing
from skimage.transform import hough_circle, hough_circle_peaks



class Pupillometry:


    def __init__(self):
        # Mode de teste
        self.test_mode = False
        self.radius_pupil_iris_test_mode = {
            '14_2_frame_0125.jpg': [(100, 100, 100), (100, 100, 100)],
            '30_2_frame_0149.jpg': [(100, 100, 100), (100, 100, 100)],
            '48_1_frame_0149.jpg': [(100, 100, 100), (100, 100, 100)]
        }

        # Constantes
        self.blur_filters = ['Média', 'Mediana', 'Gaussiano', 'Equalização de histograma']
        self.high_filters = ['Prewitt vertical', 'Sobel vertical', 'Scharr vertical']
        self.morph_operators = ['Erosão', 'Dilatação', 'Abertura', 'Fechamento']
        
        # Configuração
        sns.set_style('ticks')
        self.config = {
            'pre_proc_filter_width': 5,
            'pre_proc_filter_sigma': 1.,
            'pre_proc_filter_type': 'Equalização de histograma',
            'pre_proc_flash_l': 6,
            'pre_proc_flash_k': 1.5,
            'pre_proc_flash_width': 7,
            'seg_pup_method': 'Binarização global',
            'seg_pup_bin_nbins': 15,
            'seg_pup_bin_width': 11,
            'seg_pup_morph_type': 'Fechamento',
            'seg_pup_hough_sigma': 1.,
            'seg_pup_hough_lthresh': 50,
            'seg_pup_hough_hthresh': 120,
            'seg_iris_filter_width': 5,
            'seg_iris_filter_type': 'Scharr vertical' 
        }
        self.config_translator = {
            'pre_proc_filter_width': 'Filtragem de Suavização - Largura da máscara',
            'pre_proc_filter_sigma': 'Filtragem de Suavização - Desvio padrão σ gaussiano',
            'pre_proc_filter_type': 'Filtragem de Suavização - Método de filtragem',
            'pre_proc_flash_l': 'Tratamento de Reflexos - Parâmetro L',
            'pre_proc_flash_k': 'Tratamento de Reflexos - Parâmetro k',
            'pre_proc_flash_width': 'Tratamento de Reflexos - Largura da máscara do filtro da mediana',
            'seg_pup_method': 'Segmentação da Pupila - Método',
            'seg_pup_bin_nbins': 'Binarização Global - Número de bins',
            'seg_pup_bin_width': 'Binarização Global - Largura do elemento estruturante',
            'seg_pup_morph_type': 'Morfologia Binária Matemática - Operador',
            'seg_pup_hough_sigma': 'Detector de Bordas de Canny - Desvio padrão σ gaussiano',
            'seg_pup_hough_lthresh': 'Detector de Bordas de Canny - Limiar inferior',
            'seg_pup_hough_hthresh': 'Detector de Bordas de Canny - Limiar inferior',
            'seg_iris_filter_width': 'Filtragem de Realce - Largura da máscara',
            'seg_iris_filter_type': 'Filtragem de Realce - Método de filtragem' 
        }

        # Estado
        self.images_to_process = dict()
        self.images_hash_processed = list()
        self.radius_pupil_iris = list()
        self.artifacts_found = list()
        self.image_processing = None
        self.image_filtered_blur = None
        self.image_pre_processed = None
        self.image_pupil = None
        self.image_filtered_high = None
        self.image_iris = None
        self.image_processed = None
        self.images_to_plot_blur = None
        self.images_to_plot_morph = None
        self.images_to_plot_high = None
        self.region_pupil = None
        self.bbox_pupil = None
        self.region_iris = None
        self.bbox_iris = None


    def reset(self):
        self.artifacts_found = list()
        self.image_processing = None
        self.image_filtered_blur = None
        self.image_pre_processed = None
        self.image_pupil = None
        self.image_filtered_high = None
        self.image_iris = None
        self.image_processed = None
        self.images_to_plot_blur = None
        self.images_to_plot_morph = None
        self.images_to_plot_high = None
        self.region_pupil = None
        self.bbox_pupil = None
        self.region_iris = None
        self.bbox_iris = None


    def get_summary_config(self):
        return pd.DataFrame(
            columns=['Valor'],
            index=[self.config_translator[i] for i in list(self.config.keys())],
            data=np.array(list(self.config.values())))


    def get_image(self, image_bytes):
        return io.imread(image_bytes, as_gray=True, plugin='pil')


    def get_image_as_float(self, image):
        return img_as_float(image)


    def get_image_as_ubyte(self, image):
        return img_as_ubyte(image)


    def get_perceptual_hash(self, image):
        return imagehash.whash(Image.open(image))


    def label_to_rgb(self, image_labels, image):
        return label2rgb(image_labels, image=image, bg_label=0)


    def show_image(self, image, cmap='gray'):
        plt.imshow(image, cmap)


    def show_image_and_summary_table(self, image, cmap='gray'):
        plt.imshow(image, cmap)
        image_pre_processed_ubyte = self.get_image_as_ubyte(image)
        self.show_image(image_pre_processed_ubyte)

        return pd.DataFrame(
            columns=['Valor'],
            index=['Dimensão (linhas, colunas)', 'Valor mínimo de intensidade', 'Valor máximo de intensidade'],
            data=np.array([image_pre_processed_ubyte.shape, image_pre_processed_ubyte.min(), image_pre_processed_ubyte.max()]))


    def show_image_and_histogram(self, image, image_bin, n_bins, cmap='gray'):
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.hist(image.ravel(), bins=n_bins)
        fig.add_subplot(1, 2, 2)
        plt.imshow(image_bin, cmap)


    def show_all_images(self, images, nrows, ncols, titles=None, cmap='gray'):
        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        for ax, image, label in zip(axes.ravel(), images, titles):
            ax.imshow(image, cmap)
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
        for type_of_operation in self.morph_operators:
            images_morph[type_of_operation] = self.transform_image(image, selem, None, type_of_operation)
        return images_morph

    
    def get_high_filtered_images(self, selem, image):
        images_filtered = dict()
        for type_of_filter in self.high_filters:
            images_filtered[type_of_filter] = self.transform_image(image, selem, None, type_of_filter)
        return images_filtered


    def transform_image(self, image, selem, sigma, type_of_function):
        # Filtros de suavização
        if type_of_function == 'Média':
            return self.get_image_as_float(mean_filter(self.get_image_as_ubyte(image), selem))
        elif type_of_function == 'Mediana':
            return self.get_image_as_float(median_filter(self.get_image_as_ubyte(image), selem))
        elif type_of_function == 'Gaussiano':
            return gaussian_filter(image, sigma)
        elif type_of_function == 'Equalização de histograma':
            return equalize_hist(image)

        # Filtros de realce
        elif type_of_function == 'Prewitt horizontal':
            return prewitt_h(image)
        elif type_of_function == 'Prewitt vertical':
            return prewitt_v(image)
        elif type_of_function == 'Prewitt':
            return prewitt(image)
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


    def get_bounding_box(self, region):
        return region[0] - region[2], region[1] - region[2], region[0] + region[2], region[1] + region[2]


    def draw_circle_perimeter(self, image, region):
        r_coords_pupil, c_coords_pupil = circle_perimeter(region[0], region[1], region[2], shape=image.shape)
        image[r_coords_pupil, c_coords_pupil] = 1
        return image


    def iris_segmentation_horizontal_gradient(self, image, offset_col_low, offset_col_high):
        # col_pos = col_pupil + radius_pupil + offset_col_high
        col_pos = self.region_pupil[1] + self.region_pupil[2] + offset_col_low
        col_gradient = np.gradient(image[self.region_pupil[0], col_pos:(col_pos + offset_col_high)])

        col_edge = np.argmax(col_gradient) + col_pos
        rad = col_edge - self.region_pupil[0]

        # region = row, column, radius
        region = (self.region_pupil[0], self.region_pupil[1], rad)
        return region


    def lineplot(self):
        df = pd.DataFrame(
            columns=['Raio pupila (pixels)', 'Raio íris (pixels)'],
            index=list(range(1, len(self.radius_pupil_iris)+1)),
            data=np.array(self.radius_pupil_iris))

        sns.lineplot(data=df, markers=True, dashes=False)
        plt.xlabel('Quadro')
        plt.ylabel('Raio (pixels)')
        plt.legend(labels=['Pupila', 'Íris'])


    def get_df_results(self):
        return pd.DataFrame(
            columns=['Valor'],
            index=['Pupila - Centro (linha, coluna)', 'Pupila - Raio (pixels)', 'Íris - Centro (linha, coluna)', 'Íris - Raio (pixels)'],
            data=np.array([
                '(' + str(self.region_pupil[0]) + ', ' + str(self.region_pupil[1]) + ')', self.region_pupil[2],
                '(' + str(self.region_iris[0]) + ', ' + str(self.region_iris[1]) + ')', self.region_iris[2]
            ]))
