import numpy as np
import time

from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.morphology import disk 
from skimage.filters import scharr_v
from skimage.filters.rank import median as median_filter
from skimage.draw import circle
from skimage.measure import regionprops, label as measure_label

def change_value_neighborhood(img, pixel, L):
  r, c = pixel[0], pixel[1]
  
  # Acha as coordenadas de um círculo de raio L centrado em (x, y)
  r_coords, c_coords = circle(r, c, L, img.shape)

  # Calcula o valor mínimo entre todos os pixel localizados a uma distância menor que L 
  new_value = img[r_coords, c_coords].min()

  # Substitui o pixel e seus vizinhos pelo novo valor
  img[r_coords, c_coords] = new_value
  return

def mean_value_neighborhood(img, pixel):
  r, c = pixel[0], pixel[1]
  return np.mean([img[r, c], img[r+1, c], img[r-1, c], img[r, c+1], img[r, c-1]])

def remove_flashes(img, L=6, k=1.5):
  print('Removendo reflexos especulares...')
  
  lim_min, lim_row_max, lim_col_max, = L+1, img.shape[0]-L-1, img.shape[1]-L-1

  # Para cada pixel (x, y)
  for r in range(lim_min, lim_row_max):
    for c in range(lim_min, lim_col_max):
      # Calcula o valor médio entre o pixel e sua 4-vizinhança
      v0 = mean_value_neighborhood(img, pixel=(r, c))
     
      # Calcula o valor médio dos pixels L distantes
      vr = mean_value_neighborhood(img, pixel=(r+L, c))
      vl = mean_value_neighborhood(img, pixel=(r-L, c))
      vu = mean_value_neighborhood(img, pixel=(r, c+L))
      vd = mean_value_neighborhood(img, pixel=(r, c-L))

      # Verifica se pixel está próximo ao centro do artefato a ser removido
      change_pixel = (v0 > vr*k) and (v0 > vl*k) and (vu > vr*k) and (vd > vr*k)
      if change_pixel:
      	change_value_neighborhood(img, (r, c), L)
  
  return img

def pre_processing(image, L, k):
    img = image.copy()

    remove_flashes(img, L, k)
    flashes_removed = median_filter(img_as_ubyte(img), disk(L))

    return flashes_removed

def pupil_segmentation(image, n_bins):
    seg_pupil = image.copy()

    _, bins = np.histogram(seg_pupil.ravel(), n_bins)
    t = bins[1]

    # Binariza imagem a partir do threshold
    thresh = img_as_float(seg_pupil < t)

    # Determina as regiões conexas
    labels = measure_label(thresh, background=0)
    
    # Calcula métricas das regiões encontradas
    regions = regionprops(labels)

    row_pupil, col_pupil, radius_pupil = None, None, None
    for region in regions:
        area_pupil = region.area
        if area_pupil > 1000:
            row_pupil, col_pupil = map(int, region.centroid)
            radius_pupil = int(np.sqrt(area_pupil / np.pi))

    return row_pupil, col_pupil, radius_pupil

def iris_segmentation(row_pupil, col_pupil, radius_pupil, image, a, b):
    seg_iris = scharr_v(image.copy())
    
    offset_col = col_pupil + radius_pupil + a
    
    gradient_col = np.gradient(seg_iris[row_pupil, offset_col:offset_col+b])

    col_edge = np.argmax(gradient_col) + offset_col

    radius_iris = col_edge - col_pupil

    return radius_iris

def segmentation(filename):
    start = time.time() 

    # Carrega a imagem
    original = rgb2gray(io.imread(filename))

    # Reduz reflexos especulares na imagem
    flashes_removed = pre_processing(image=original, L=10, k=1.15)

    # Faz a segmentação da pupila
    row_pupil, col_pupil, radius_pupil = pupil_segmentation(image=flashes_removed, n_bins=25)

    # Considerando que a íris e a pupila são concêntricas
    if radius_pupil is None:
      radius_iris = None
    else:
      radius_iris = iris_segmentation(row_pupil, col_pupil, radius_pupil, 
                                        image=flashes_removed, a=10, b=100)
    
    print(f'Tempo de processamento: {time.time()-start} seg')

    return row_pupil, col_pupil, radius_pupil, radius_iris
