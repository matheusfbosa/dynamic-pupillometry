import streamlit as st
import os
import pandas as pd
from typing import Dict

from pupillometry import Pupillometry



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_pupillometry():
    return Pupillometry()


pup = get_pupillometry()


def main():
    with open(os.path.join(os.path.dirname(__file__), 'styles.css')) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.sidebar.title('Pupilometria Dinâmica')
    st.title('Pupilometria Dinâmica :o:')
    st.sidebar.info('Aplicação que realiza a segmentação das regiões da pupila e íris em imagens de pupilometria dinâmica.')

    pup = get_pupillometry()

    load_image()
    
    mode = st.sidebar.radio('Modo de execução:', ('Completo', 'Passo a passo'))

    if mode == 'Completo':
        if len(pup.images_in) > 0:
            if st.button('Segmentar pupila e íris'):
                run()
            
    elif mode == 'Passo a passo':
        st.sidebar.markdown('Etapas:')

        if st.sidebar.checkbox('Pré-Processamento'):
            if len(pup.images_in) > 0:
                pre_processing()
            else:
                st.warning('Por favor, execute a etapa de carregamento de imagem antes de proceder.')

        if st.sidebar.checkbox('Segmentação da pupila'):
            if pup.image_pre_processed is not None:
                pupil_segmentation()
            else:
                st.warning('Por favor, execute a etapa de pré-processamento antes de proceder.')

        if st.sidebar.checkbox('Segmentação da íris'):
            if pup.region_pupil is not None:
                iris_segmentation()
            else:
                st.warning('Por favor, execute a etapa de segmentação da pupila antes de proceder.')

        if st.sidebar.checkbox('Resultados'):
            if pup.region_iris is not None:
                results()
            else:
                st.warning('Por favor, execute a etapa de segmentação da íris antes de proceder.')

    about()


def multiple_files_uploader():
    image_bytes = st.file_uploader('Upload de imagens:', type=['jpg', 'png'])
    if image_bytes:
        # Gera hash da imagem para verificações
        hashfile = pup.get_perceptual_hash(image_bytes)

        # Adiciona imagem ao dicionário caso hash já não seja uma chave
        if not hashfile in pup.images_in.keys():
            pup.images_in[hashfile] = pup.get_image(image_bytes)
    else:
        pup.images_in.clear() # Limpa dicionário caso usuário limpe cache e recarregue página
        st.info('Realize o upload de uma ou mais imagens do tipo .jpg ou .png.')

    if len(pup.images_in.keys()) > 0: 
        if st.button('Limpar lista de imagens'):
            pup.images_in.clear()
        if st.checkbox('Mostrar lista de imagens', True):
            for index, hashfile in enumerate(pup.images_in.keys()):
                st.text(str(index+1) + ' - hash ' + str(hashfile))
                if st.checkbox('Mostrar imagem ' + str(index+1)):
                    image_pre_processed_ubyte = pup.get_image_as_ubyte(pup.images_in.get(hashfile))
                    pup.show_image(image_pre_processed_ubyte)
                    st.pyplot()

                    df = pd.DataFrame(
                        index=['Tamanho', 'Valor mínimo', 'Valor máximo'],
                        columns=['Valor'],
                        data=pup.get_np_array([image_pre_processed_ubyte.shape, image_pre_processed_ubyte.min(), image_pre_processed_ubyte.max()])
                    )
                    st.table(df)
    

def load_image():
    st.subheader(':open_file_folder: Carregar imagem')

    option_image = st.radio('Selecione:', ('Realizar upload de imagens', 'Utilizar imagem de exemplo'))

    if option_image == 'Realizar upload de imagens':
        multiple_files_uploader()

    elif option_image == 'Utilizar imagem de exemplo':
        filenames = os.listdir(os.path.join(os.path.dirname(__file__), 'images'))
        filename_selected = st.selectbox('Selecione:', tuple(filenames))
        fname = os.path.join(os.path.dirname(__file__), 'images', filename_selected)
        hashfile = pup.get_perceptual_hash(fname)

        if not hashfile in pup.images_in.keys():
            pup.images_in.clear()
            pup.images_in[hashfile] = pup.get_image(fname)

        if pup.images_in is not None:
            if st.checkbox('Mostrar imagem'):
                image_pre_processed_ubyte = pup.get_image_as_ubyte(next(iter(pup.images_in.values())))
                pup.show_image(image_pre_processed_ubyte)
                st.pyplot()

                df = pd.DataFrame(
                    index=['Tamanho', 'Valor mínimo', 'Valor máximo'],
                    columns=['Valor'],
                    data=pup.get_np_array([image_pre_processed_ubyte.shape, image_pre_processed_ubyte.min(), image_pre_processed_ubyte.max()])
                )
                st.table(df)
    

def run():
    # Pré-Processamento
    type_of_blur_filter = 'Equalização de histograma'
    width_pre = 5
    L = 12
    k = 1.3

    pup.image_filtered = pup.transform_image(image=pup.image_in, selem=pup.get_selem(width_pre), sigma=None, type_of_function=type_of_blur_filter)
    pup.image_pre_processed, pup.artifacts_found = __handle_flashes(L, k, pup.get_selem(width_pre))

    # Segmentação da pupila
    n_bins = 15
    width_morph = 11
    type_of_morph_operator = 'Fechamento'

    _, bins = pup.get_histogram(image=pup.image_pre_processed, n_bins=n_bins)
    thresh = int(round(bins[1]))
    image_bin = pup.get_image_as_float(pup.image_pre_processed < thresh)
    image_morph = pup.transform_image(image=image_bin, selem=pup.get_selem(width_morph), sigma=None, type_of_function=type_of_morph_operator)
    image_labels, _ = pup.labelizer_image(image=image_morph)
    regions = pup.get_regionprops(image_labels)
    region = max(regions, key=lambda item: item.area)
    row_pupil, col_pupil = map(int, region.centroid)
    radius_pupil = region.equivalent_diameter // 2
    pup.region_pupil = (row_pupil, col_pupil, radius_pupil)
    pup.image_pupil = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_pupil)

    # Segmentação da íris
    pup.region_iris = __iris_segmentation_horizontal_gradient()
    pup.image_iris = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_iris)

    # Resultados
    results()


def pre_processing():
    st.subheader(':wrench: Pré-Processamento')
        
    if st.checkbox('Filtragem'):
        if pup.image_in is not None:
            width = st.slider('Largura do elemento estruturante:', min_value=1, max_value=20, value=3)
            selem = pup.get_selem(width)
            st.write(selem)
            sigma = st.slider('Valor do desvio padrão σ do kernel gaussiano:', min_value=.1, max_value=5., value=1.)
            
            if st.button('Filtrar'):
                pup.images_to_plot_blur = __images_to_plot_blur_filtering(selem, sigma)

            if pup.images_to_plot_blur is not None:
                st.markdown('Resultado:')
                pup.show_all_images(pup.images_to_plot_blur.values(), nrows=2, ncols=2, titles=pup.blur_filters)
                st.pyplot()
                
                type_of_filter_selected = st.selectbox('Selecione o filtro com melhor resultado:', pup.blur_filters)
                pup.image_filtered = pup.images_to_plot_blur.get(type_of_filter_selected)
            
            else:
                st.info('Configure os parâmetros e clique sobre o botão "Filtrar".')

        else:
            st.warning('Por favor, execute a etapa de carregamento de imagem antes de proceder.')
        
    if st.checkbox('Tratamento de reflexos'):
        if pup.image_filtered is not None:
            image_flash_parameters = os.path.join(os.path.dirname(__file__), 'flash_parameters.png')
            image_flash_example = os.path.join(os.path.dirname(__file__), 'images', '12_2_frame_0000.jpg')
            st.image(image=[image_flash_example, image_flash_parameters], caption=['Reflexos numa imagem.', 'Parâmetros L e k.'], width=300)
            #image_flash_parameters = pup.get_image(os.path.join(os.path.dirname(__file__), 'flash_parameters.png'))
            #image_flash_example = pup.get_image(os.path.join(os.path.dirname(__file__), 'images', '12_2_frame_0000.jpg'))
            #pup.show_all_images([image_flash_parameters, image_flash_example], nrows=1, ncols=2, titles=['Reflexos em imagem', 'Parâmetros L e k'])
            #st.pyplot()

            L = st.slider('Valor do parâmetro L:', min_value=1, max_value=50, value=12)
            k = st.slider('Valor do parâmetro k:', min_value=1.1, max_value=5.0, value=1.3)
            width = st.slider('Largura do elemento estruturante para filtragem de suavização:', min_value=1, max_value=20, value=3)
            selem = pup.get_selem(width)
            st.write(selem)

            if st.button('Tratar reflexos'):
                st.info('Tratando reflexos. Isso pode demorar um pouco...')
                pup.image_pre_processed, pup.artifacts_found = __handle_flashes(L, k, selem)

            if len(pup.artifacts_found) > 0:
                pup.show_all_images([pup.image_filtered, pup.image_pre_processed], titles=['Antes', 'Depois'], nrows=1, ncols=2)
                st.pyplot()
                st.success('Reflexos encontrados e tratados em: ' + str(pup.artifacts_found).strip('[]') + '.')
            elif pup.image_pre_processed is None:
                st.info('Configure os parâmetros e clique sobre o botão "Tratar reflexos".')
            else:
                st.success('Nenhum reflexo foi encontrado a partir dos parâmetros.')
        else:
            st.warning('Por favor, execute a etapa de filtragem antes de proceder.')


@st.cache(suppress_st_warning=True)
def __images_to_plot_blur_filtering(selem, sigma):
    return pup.get_blur_filtered_images(selem, sigma, image=pup.image_in)


@st.cache(suppress_st_warning=True)
def __handle_flashes(L, k, selem):
    image = pup.image_filtered.copy()
    artifacts = pup.handle_flashes(L, k, image)
    pup.image_pre_processed = pup.transform_image(image, selem=selem, sigma=None, type_of_function='Mediana')
    return artifacts


def pupil_segmentation():
    st.subheader(':black_circle: Segmentação da pupila')

    option_method = st.radio('Selecione um método:', ('Binarização global', 'Detecção de bordas e transformada circular de Hough'))

    if option_method == 'Binarização global':
        pupil_segment_global_binarization()

    elif option_method == 'Detecção de bordas e transformada circular de Hough':
        pupil_segment_canny_hough()

    if pup.region_pupil is not None:
        st.markdown(':pushpin: Pupila segmentada')
        pup.image_pupil = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_pupil)
        pup.show_image(pup.image_pupil)
        st.pyplot()

        st.success('Pupila encontrada em ('
                    + str(pup.region_pupil[0]) + ', ' 
                    + str(pup.region_pupil[1]) + ') com raio de '
                    + str(pup.region_pupil[2]) + ' pixels.')

    else:
        st.warning('A região da pupila não foi encontrada na imagem.')
        pup.image_pupil = None


def pupil_segment_global_binarization():
    n_bins = st.slider('Número de bins do histograma:', min_value=0, max_value=100, value=15)

    image_pre_processed_ubyte = pup.get_image_as_ubyte(pup.image_pre_processed)
    hist, bins = pup.get_histogram(image=image_pre_processed_ubyte, n_bins=n_bins)
    thresh = int(round(bins[1]))
    
    image_bin = pup.get_image_as_float(image_pre_processed_ubyte < thresh)

    st.text('Valor do limiar de binarização: ' + str(thresh))
    pup.show_image_and_histogram(image=image_pre_processed_ubyte, image_bin=image_bin, n_bins=n_bins)
    st.pyplot()

    st.markdown(':pushpin: Morfologia matemática binária')
    width = st.slider('Largura do elemento estruturante morfológico:', min_value=1, max_value=20, value=3)
    selem = pup.get_selem(width)
    st.write(selem)
    
    if st.button('Aplicar operações morfológicas'):
        pup.images_to_plot_morph = __images_to_plot_morphology(selem, image_bin)

    if pup.images_to_plot_morph is not None:
        st.markdown('Resultado:')
        pup.show_all_images(pup.images_to_plot_morph.values(), titles=pup.morphological_operators, nrows=2, ncols=2)
        st.pyplot()
        
        type_of_operator_selected = st.selectbox('Selecione a operação com melhor resultado:', pup.morphological_operators)
        image_morph = pup.images_to_plot_morph.get(type_of_operator_selected)
    
        image_labels, n_labels = pup.labelizer_image(image=image_morph)

        st.write('Número de regiões encontradas: ' + str(n_labels))
        if st.checkbox('Mostrar imagem com rótulos'):
            image_labels_colored = pup.label_to_rgb(image_labels, image=pup.image_pre_processed)
            pup.show_image(image_labels_colored)
            st.pyplot()

        regions = pup.get_regionprops(image_labels)
        region = max(regions, key=lambda item: item.area)
        row_pupil, col_pupil = map(int, region.centroid)
        radius_pupil = region.equivalent_diameter // 2
        #min_row, min_col, max_row, max_col = region.bbox

        pup.region_pupil = (row_pupil, col_pupil, radius_pupil)

    else:
        st.info('Configure os parâmetros e clique sobre o botão "Aplicar operações morfológicas".')


@st.cache(suppress_st_warning=True)
def __images_to_plot_morphology(selem, image_bin):
    return pup.get_images_morphologically_transformed(selem, image=image_bin)


def pupil_segment_canny_hough():
    st.markdown(':pushpin: Filtro de Canny')
    sigma = st.slider('Valor do desvio padrão σ:', min_value=0.1, max_value=5.0, value=1.0)
    low_threshold = 0.0
    high_threshold = 0.0

    option_thresh = st.radio('Configuração dos valores de limiar:', ('Manual', 'Otsu'))
    
    if option_thresh == 'Manual':
        low_threshold = st.slider('Valor do limiar inferior:', min_value=0, max_value=255, value=1)
        high_threshold = st.slider('Valor do limiar superior:', min_value=0, max_value=255, value=10)
    
    elif option_thresh == 'Otsu':
        otsu_thresh = pup.get_otsu_threshold(image=pup.image_pre_processed)
        low_threshold = 0.5 * otsu_thresh
        high_threshold = otsu_thresh
        st.text('Limiar inferior: ' + str(low_threshold) + ' (0,5 * Otsu)')
        st.text('Limiar superior: ' + str(high_threshold) + ' (Otsu)')

    st.info('Detectando bordas...')
    image_edges = __canny_filter(sigma, low_threshold, high_threshold, image=pup.image_pre_processed)
    st.success('Bordas detectadas.')
    pup.show_image(image_edges)
    st.pyplot()

    st.markdown(':pushpin: Transformada circular de Hough')
    st.info('Buscando por circunferências...')
    pup.region_pupil = __circle_hough_transform(image_edges)


@st.cache(suppress_st_warning=True)
def __canny_filter(sigma, low_threshold, high_threshold, image):
    return pup.get_edges_canny(sigma, low_threshold, high_threshold, image.copy())


@st.cache(suppress_st_warning=True)
def __circle_hough_transform(image_edges):
    return pup.pupil_segmentation_hough(image_edges.copy())


def iris_segmentation():
    st.subheader(':o: Segmentação da íris')

    st.info('Buscando pela região de fronteira entre íris e esclera..')
    pup.region_iris = __iris_segmentation_horizontal_gradient()

    if pup.region_iris is not None:
        st.markdown(':pushpin: Íris segmentada')

        pup.image_iris = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_iris)
        pup.show_image(pup.image_iris)
        st.pyplot()

        st.success('Íris encontrada em ('
                    + str(pup.region_iris[0]) + ', ' 
                    + str(pup.region_iris[1]) + ') com raio de '
                    + str(pup.region_iris[2]) + ' pixels.')

    else:
        st.warning('A região da íris não foi encontrada na imagem.')
        pup.image_iris = None
   

@st.cache(suppress_st_warning=True)
def __images_to_plot_highlight_filtering(selem):
    return pup.get_highlight_filtered_images(selem, image=pup.image_pupil)


def __iris_segmentation_horizontal_gradient():
    return pup.iris_segmentation_horizontal_gradient(image=pup.image_pre_processed.copy())


def results():
    st.subheader(':bar_chart: Resultados')

    df = pd.DataFrame(
        index=['Pupila - Posição (x, y)', 'Pupila - Raio (pixels)', 'Íris - Posição (x, y)', 'Íris - Raio (pixels)'],
        columns=['Valor'],
        data=pup.get_np_array([
            '(' + str(pup.region_pupil[0]) + ', ' + str(pup.region_pupil[1]) + ')', pup.region_pupil[2],
            '(' + str(pup.region_iris[0]) + ', ' + str(pup.region_iris[1]) + ')', pup.region_iris[2]
        ])
    )
    st.table(df)


def about():
    st.sidebar.title('Sobre')
    st.sidebar.info(
        '''
        Desenvolvido por **Matheus Bosa** como requisito para avaliação em TE105 Projeto de Graduação
        do curso de Engenharia Elétrica da Universidade Federal do Paraná (UFPR).
        [LinkedIn](https://www.linkedin.com/in/matheusbosa/) e [GitHub](https://github.com/bosamatheus).
        '''
    )
    st.sidebar.image(os.path.join(os.path.dirname(__file__), 'profile.jpg'), use_column_width=True)


if __name__ == '__main__':
    main()
