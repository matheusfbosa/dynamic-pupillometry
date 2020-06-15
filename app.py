import streamlit as st
import os
import time

from pupillometry import Pupillometry



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_pupillometry():
    """Singleton da classe Pupillometry.

    Returns:
        Pupillometry: classe com configurações e operações sobre imagens de pupilometria.
    """
    return Pupillometry()


pup = get_pupillometry()


def main():
    # Carrega arquivo CSS com estilos
    with open(os.path.join(os.path.dirname(__file__), 'resources', 'styles.css')) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.sidebar.title('Pupilometria Dinâmica')
    st.sidebar.image('./resources/ufpr.jpg', width=150)
    st.sidebar.image('./resources/delt.png', width=100)
    st.title('Pupilometria Dinâmica :o:')
    st.sidebar.info('Sistema de detecção automática de pupila e íris em imagens de pupilometria dinâmica.')

    pup = get_pupillometry()

    load_image()
    
    mode = st.sidebar.radio('Modo de execução:', ('Completo', 'Configuração'))

    if mode == 'Completo':
        if len(pup.images_to_process) > 0:
            if st.checkbox('Mostrar configuração'):
                st.subheader(':mag_right: Configuração')
                st.write(pup.get_summary_config())

            if st.checkbox('Segmentar pupila e íris'):
                st.info(f'Processando quadro {len(pup.images_hash_processed)+1} de {len(pup.images_to_process)}.')
                for hashfile, image in pup.images_to_process.items():
                    if hashfile not in pup.images_hash_processed:
                        pup.image_processing = image
                        run_all_steps()
                        pup.images_hash_processed.append(hashfile)
                plot_radius_chart()
            else:
                pup.images_hash_processed = list()

    elif mode == 'Configuração':
        st.sidebar.markdown('Etapas:')

        if st.sidebar.checkbox('Pré-Processamento'):
            if pup.image_processing is not None:
                pre_processing()
            else:
                st.warning('Execute a etapa de carregamento de imagem antes de proceder.')

        if st.sidebar.checkbox('Segmentação da pupila'):
            if pup.image_pre_processed is not None:
                pupil_segmentation()
            else:
                st.warning('Execute a etapa de pré-processamento antes de proceder.')

        if st.sidebar.checkbox('Segmentação da íris'):
            if pup.region_pupil is not None:
                iris_segmentation()
            else:
                st.warning('Execute a etapa de segmentação da pupila antes de proceder.')

        if st.sidebar.checkbox('Resultados'):
            if pup.region_iris is not None:
                results()
            else:
                st.warning('Execute a etapa de segmentação da íris antes de proceder.')

    about()


def load_image():
    st.subheader(':open_file_folder: Carregar imagem')

    option_image = st.radio('Selecione:', ('Realizar upload de imagens', 'Utilizar imagem de exemplo'))

    if option_image == 'Realizar upload de imagens':
        multiple_files_uploader()

        if len(pup.images_to_process) > 0:
            hashfile = st.selectbox('Selecione uma das imagens carregadas:', tuple(pup.images_to_process.keys()))
            pup.image_processing = pup.images_to_process.get(hashfile)
        else:
            st.warning('Selecione uma das imagens carregadas antes de proceder.')

    elif option_image == 'Utilizar imagem de exemplo':
        filenames = os.listdir(os.path.join(os.path.dirname(__file__), 'images'))
        filename_selected = st.selectbox('Selecione:', tuple(filenames))
        fname = os.path.join(os.path.dirname(__file__), 'images', filename_selected)
        hashfile = pup.get_perceptual_hash(fname)

        if not hashfile in pup.images_to_process.keys():
            pup.images_to_process.clear()
            pup.images_to_process[hashfile] = pup.get_image(fname)
            pup.image_processing = pup.images_to_process.get(hashfile)

        if pup.image_processing is not None:
            if st.checkbox('Mostrar imagem'):
                df = pup.show_image_and_summary_table(image=pup.get_image_as_ubyte(next(iter(pup.images_to_process.values()))))
                st.pyplot()
                st.table(df)
    

def multiple_files_uploader():
    image_bytes = st.file_uploader('Upload de imagens:', type=['jpg', 'png'])
    if image_bytes:
        # Gera hash da imagem para verificações
        hashfile = pup.get_perceptual_hash(image_bytes)

        # Adiciona imagem ao dicionário caso hash já não seja uma chave
        if not hashfile in pup.images_to_process.keys():
            pup.images_to_process[hashfile] = pup.get_image(image_bytes)
    else:
        # Limpa dicionário caso usuário limpe cache e recarregue página
        pup.images_to_process.clear()
        st.info('Realize o upload de uma ou mais imagens do tipo .jpg ou .png.')

    if len(pup.images_to_process.keys()) > 0: 
        if st.button('Limpar lista de imagens'):
            pup.images_to_process.clear()
        if st.checkbox('Mostrar lista de imagens', True):
            for index, hashfile in enumerate(pup.images_to_process.keys()):
                st.text(f'{index+1} - hash {hashfile}')
                if st.checkbox('Mostrar imagem ' + str(index+1)):
                    df = pup.show_image_and_summary_table(image=pup.images_to_process.get(hashfile))
                    st.pyplot()
                    st.table(df)


def run_all_steps():
    start_time = time.time()

    # ---> Pré-Processamento <----
    # Filtragem de suavização
    selem_blur_filter = pup.get_selem(pup.config['pre_proc_filter_width'])
    pup.image_filtered_blur = pup.transform_image(image=pup.image_processing.copy(), 
                                    selem=selem_blur_filter, 
                                    sigma=pup.config['pre_proc_filter_sigma'],
                                    type_of_function=pup.config['pre_proc_filter_type'])

    # Tratamento de reflexos 
    st.info('Tratando reflexos. Isso pode demorar um pouco...')
    pup.image_pre_processed, pup.artifacts_found = __handle_flashes(L=pup.config['pre_proc_flash_l'], 
                                                        k=pup.config['pre_proc_flash_k'], 
                                                        selem=selem_blur_filter)
    if len(pup.artifacts_found) > 0:
        st.success(f'Reflexos encontrados e tratados em: {str(pup.artifacts_found).strip("[]")}.')

    # ---> Segmentação da pupila <----
    image_pre_processed_ubyte = pup.get_image_as_ubyte(pup.image_pre_processed)

    if pup.config['seg_pup_method'] == 'Binarização global':
        # Histograma
        _, bins = pup.get_histogram(image=image_pre_processed_ubyte, n_bins=pup.config['seg_pup_bin_nbins'])
        thresh = int(round(bins[1]))
        st.text(f'Valor do limiar de binarização: {thresh}')
        image_bin = pup.get_image_as_float(image_pre_processed_ubyte < thresh)

        # Morfologia matemática
        image_morph = pup.transform_image(image=image_bin.copy(),
                                    selem=pup.get_selem(pup.config['seg_pup_bin_width']),
                                    sigma=None,
                                    type_of_function=pup.config['seg_pup_morph_type'])
        image_labels, n_labels = pup.labelizer_image(image=image_morph)
        st.text(f'Número de regiões encontradas: {n_labels}')

        # Localização da pupila
        regions = pup.get_regionprops(image_labels)
        region = max(regions, key=lambda item: item.area)
        row_pupil, col_pupil = map(int, region.centroid)
        radius_pupil = int(region.equivalent_diameter // 2)
        pup.region_pupil = (row_pupil, col_pupil, radius_pupil)
        pup.bbox_pupil = region.bbox
        
    elif pup.config['seg_pup_method'] == 'Identificação de circunferências':
        image_edges = __canny_filter(pup.config['seg_pup_hough_sigma'], pup.config['seg_pup_hough_lthresh'], pup.config['seg_pup_hough_hthresh'], image=image_pre_processed_ubyte)
        
        st.info('Aplicando transformada de Hough...')
        pup.region_pupil = __circle_hough_transform(image_edges)
        pup.bbox_pupil = pup.get_bounding_box(region=pup.region_pupil)
        st.success('Busca por circunferências na imagem foi encerrada.')

    # ---> Segmentação da íris <----
    pup.image_filtered_high = pup.transform_image(image=pup.image_pre_processed.copy(),
                                selem=pup.get_selem(pup.config['seg_pup_bin_width']),
                                sigma=None,
                                type_of_function=pup.config['seg_iris_filter_type'])
    st.info('Buscando pela região de fronteira entre íris e esclera..')
    pup.region_iris = __iris_segmentation_horizontal_gradient(image=pup.image_filtered_high)
    pup.bbox_iris = pup.get_bounding_box(region=pup.region_iris)
    st.success('Busca encerrada.')

    st.text(f'Tempo de processamento do quadro: {round(time.time() - start_time, 3)} segundos.')

    # ---> Resultados <----
    results()


def pre_processing():
    st.subheader(':wrench: Pré-Processamento')
    
    if st.checkbox('Filtragem de suavização'):
        if pup.image_processing is not None:
            pup.config['pre_proc_filter_width'] = st.slider('Largura do elemento estruturante:', min_value=1, max_value=20, value=5)
            selem = pup.get_selem(pup.config['pre_proc_filter_width'])
            st.write(selem)
            pup.config['pre_proc_filter_sigma'] = st.slider('Valor do desvio padrão σ do kernel gaussiano:', min_value=.01, max_value=10., value=1.)
    
            if st.button('Filtrar'):
                pup.images_to_plot_blur = __images_to_plot_blur_filtering(selem, pup.config['pre_proc_filter_sigma'])

            if pup.images_to_plot_blur is not None:
                st.markdown('Resultado:')
                pup.show_all_images(pup.images_to_plot_blur.values(), nrows=2, ncols=2, titles=pup.blur_filters)
                st.pyplot()
                
                pup.config['pre_proc_filter_type'] = st.selectbox('Selecione o filtro com melhor resultado:', pup.blur_filters)
                pup.image_filtered_blur = pup.images_to_plot_blur.get(pup.config['pre_proc_filter_type'])

            else:
                st.info('Configure os parâmetros e clique sobre o botão "Filtrar".')

        else:
            st.warning('Execute a etapa de carregamento de imagem antes de proceder.')
        
    if st.checkbox('Tratamento de reflexos'):
        if pup.image_filtered_blur is not None:
            st.image(image='./resources/flash_parameters.png', caption='Parâmetros L e k.', width=200)

            pup.config['pre_proc_flash_l'] = st.slider('Valor do parâmetro L:', min_value=1, max_value=50, value=6)
            pup.config['pre_proc_flash_k'] = st.slider('Valor do parâmetro k:', min_value=1.1, max_value=5.0, value=1.5)
            pup.config['pre_proc_flash_width'] = st.slider('Largura do elemento estruturante para filtragem de suavização:', min_value=1, max_value=20, value=3)
            selem = pup.get_selem(pup.config['pre_proc_flash_width'])
            st.write(selem)

            if st.button('Tratar reflexos'):
                st.info('Tratando reflexos. Isso pode demorar um pouco...')
                pup.image_pre_processed, pup.artifacts_found = __handle_flashes(pup.config['pre_proc_flash_l'], pup.config['pre_proc_flash_k'], selem)

            if len(pup.artifacts_found) > 0:
                pup.show_all_images([pup.image_filtered_blur, pup.image_pre_processed], titles=['Antes', 'Depois'], nrows=1, ncols=2)
                st.pyplot()
                st.success(f'Reflexos encontrados e tratados em: {str(pup.artifacts_found).strip("[]")}.')

            elif pup.image_pre_processed is None:
                st.info('Configure os parâmetros e clique sobre o botão "Tratar reflexos".')

            else:
                st.success('Nenhum reflexo foi encontrado a partir dos parâmetros.')

        else:
            st.warning('Execute a etapa de filtragem antes de proceder.')


@st.cache(suppress_st_warning=True)
def __images_to_plot_blur_filtering(selem, sigma):
    return pup.get_blur_filtered_images(selem, sigma, image=pup.image_processing.copy())


@st.cache(suppress_st_warning=True)
def __handle_flashes(L, k, selem):
    image_handle_flashes, artifacts = pup.handle_flashes(L, k, pup.image_filtered_blur.copy())
    image_pre_processed = pup.transform_image(image_handle_flashes, selem=selem, sigma=None, type_of_function='Mediana')
    return image_pre_processed, artifacts


def pupil_segmentation():
    st.subheader(':black_circle: Segmentação da pupila')

    pup.config['seg_pup_method'] = st.radio('Selecione um método:', ('Binarização global', 'Identificação de circunferências'))

    if pup.config['seg_pup_method'] == 'Binarização global':
        pupil_segment_global_binarization()

    elif pup.config['seg_pup_method'] == 'Identificação de circunferências':
        pupil_segment_canny_hough()

    if pup.region_pupil is not None:
        st.markdown(':pushpin: Pupila segmentada')
        pup.image_pupil = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_pupil)
        pup.show_image(pup.image_pupil)
        st.pyplot()

        st.success(f'Pupila encontrada em ({pup.region_pupil[0]}, {pup.region_pupil[1]}) com raio de {pup.region_pupil[2]} pixels.')
        st.info(f'Bounding box (min_row, min_col, max_row, max_col): {pup.bbox_pupil}')

    else:
        st.warning('A região da pupila não foi encontrada na imagem.')
        pup.image_pupil = None


def pupil_segment_global_binarization():
    pup.config['seg_pup_bin_nbins'] = st.slider('Número de bins do histograma:', min_value=1, max_value=100, value=15)

    image_pre_processed_ubyte = pup.get_image_as_ubyte(pup.image_pre_processed)
    hist, bins = pup.get_histogram(image=image_pre_processed_ubyte, n_bins=pup.config['seg_pup_bin_nbins'])
    thresh = int(round(bins[1]))
    
    image_bin = pup.get_image_as_float(image_pre_processed_ubyte < thresh)

    st.text('Valor do limiar de binarização: ' + str(thresh))
    pup.show_image_and_histogram(image=image_pre_processed_ubyte, image_bin=image_bin, n_bins=pup.config['seg_pup_bin_nbins'])
    st.pyplot()

    st.markdown(':pushpin: Morfologia matemática binária')
    pup.config['seg_pup_bin_width'] = st.slider('Largura do elemento estruturante morfológico:', min_value=1, max_value=20, value=11)
    selem = pup.get_selem(pup.config['seg_pup_bin_width'])
    st.write(selem)
    
    if st.button('Aplicar operações morfológicas'):
        pup.images_to_plot_morph = __images_to_plot_morph(selem, image_bin)

    if pup.images_to_plot_morph is not None:
        st.markdown('Resultado:')
        pup.show_all_images(pup.images_to_plot_morph.values(), titles=pup.morph_operators, nrows=2, ncols=2)
        st.pyplot()
        
        pup.config['seg_pup_morph_type'] = st.selectbox('Selecione a operação com melhor resultado:', pup.morph_operators)
        image_morph = pup.images_to_plot_morph.get(pup.config['seg_pup_morph_type'])
    
        image_labels, n_labels = pup.labelizer_image(image=image_morph)

        st.text(f'Número de regiões encontradas: {n_labels}')
        if st.checkbox('Mostrar imagem com rótulos'):
            image_labels_colored = pup.label_to_rgb(image_labels, image=pup.image_pre_processed)
            pup.show_image(image_labels_colored)
            st.pyplot()

        regions = pup.get_regionprops(image_labels)
        region = max(regions, key=lambda item: item.area)
        row_pupil, col_pupil = map(int, region.centroid)
        radius_pupil = int(region.equivalent_diameter // 2)

        pup.region_pupil = (row_pupil, col_pupil, radius_pupil)
        pup.bbox_pupil = region.bbox

    else:
        st.info('Configure os parâmetros e clique sobre o botão "Aplicar operações morfológicas".')


@st.cache(suppress_st_warning=True)
def __images_to_plot_morph(selem, image_bin):
    return pup.get_images_morphologically_transformed(selem, image=image_bin)


def pupil_segment_canny_hough():
    st.markdown(':pushpin: Filtro de Canny')
    pup.config['seg_pup_hough_sigma'] = st.slider('Valor do desvio padrão σ:', min_value=.01, max_value=10., value=1.0)

    pup.config['seg_pup_hough_opthresh'] = st.radio('Configuração dos valores de limiar:', ('Manual', 'Otsu'))

    image_pre_processed_ubyte = pup.get_image_as_ubyte(pup.image_pre_processed)

    if pup.config['seg_pup_hough_opthresh'] == 'Manual':
        pup.config['seg_pup_hough_lthresh'] = st.slider('Valor do limiar inferior:', min_value=0, max_value=255, value=1)
        pup.config['seg_pup_hough_hthresh'] = st.slider('Valor do limiar superior:', min_value=0, max_value=255, value=10)
    
    elif pup.config['seg_pup_hough_opthresh'] == 'Otsu':
        otsu_thresh = pup.get_otsu_threshold(image=image_pre_processed_ubyte)
        pup.config['seg_pup_hough_lthresh'] = 0.5 * otsu_thresh
        pup.config['seg_pup_hough_hthresh'] = otsu_thresh
        st.text(f'Limiar inferior: {round(pup.config["seg_pup_hough_lthresh"], 3)} (0,5 * Otsu)')
        st.text(f'Limiar superior: {round(pup.config["seg_pup_hough_hthresh"], 3)} (Otsu)')

    st.info('Buscando por bordas na imagem...')
    image_edges = __canny_filter(pup.config['seg_pup_hough_sigma'], pup.config['seg_pup_hough_lthresh'], pup.config['seg_pup_hough_hthresh'], image=image_pre_processed_ubyte)
    st.success('Busca encerrada.')
    pup.show_image(image_edges)
    st.pyplot()

    st.markdown(':pushpin: Transformada circular de Hough')
    st.info('Buscando por circunferências...')
    pup.region_pupil = __circle_hough_transform(image_edges)
    pup.bbox_pupil = pup.get_bounding_box(region=pup.region_pupil)
    st.success('Busca encerrada.')


@st.cache(suppress_st_warning=True)
def __canny_filter(sigma, low_threshold, high_threshold, image):
    return pup.get_edges_canny(sigma, low_threshold, high_threshold, image.copy())


@st.cache(suppress_st_warning=True)
def __circle_hough_transform(image_edges):
    return pup.pupil_segmentation_hough(image_edges.copy())


def iris_segmentation():
    st.subheader(':o: Segmentação da íris')

    if st.checkbox('Filtragem de realce'):

        pup.config['seg_iris_filter_width'] = st.slider('Largura do elemento estruturante:', min_value=1, max_value=20, value=5)
        selem = pup.get_selem(pup.config['seg_iris_filter_width'])
        st.write(selem)

        if st.button('Filtrar'):
            pup.images_to_plot_high = __images_to_plot_high_filtering(selem)

        if pup.images_to_plot_high is not None:
            st.markdown('Resultado:')
            pup.show_all_images(pup.images_to_plot_high.values(), nrows=3, ncols=3, titles=pup.high_filters)
            st.pyplot()
            
            pup.config['seg_iris_filter_type'] = st.selectbox('Selecione o filtro com melhor resultado:', pup.high_filters)
            pup.image_filtered_high = pup.images_to_plot_high.get(pup.config['seg_iris_filter_type'])

        if pup.image_filtered_high is not None:
            st.info('Buscando pela região de fronteira entre íris e esclera..')
            pup.region_iris = __iris_segmentation_horizontal_gradient(image=pup.image_filtered_high)
            pup.bbox_iris = pup.get_bounding_box(region=pup.region_iris)
            st.success('Busca encerrada.')
        
            if pup.region_iris is not None:
                st.markdown(':pushpin: Íris segmentada')

                pup.image_iris = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_iris)
                pup.show_image(pup.image_iris)
                st.pyplot()

                st.success(f'Íris encontrada em ({pup.region_iris[0]}, {pup.region_iris[1]}) com raio de {pup.region_iris[2]} pixels.')
                st.info('Bounding box (min_row, min_col, max_row, max_col): ' + str(pup.bbox_iris))

            else:
                st.warning('A região da íris não foi encontrada na imagem.')
                pup.image_iris = None

        else:
            st.info('Configure os parâmetros e clique sobre o botão "Filtrar".')
   

@st.cache(suppress_st_warning=True)
def __images_to_plot_high_filtering(selem):
    return pup.get_high_filtered_images(selem, image=pup.image_pre_processed.copy())


def __iris_segmentation_horizontal_gradient(image):
    return pup.iris_segmentation_horizontal_gradient(image=image.copy(), offset_col_low=50, offset_col_high=120)


def results():
    st.subheader(':bar_chart: Resultados')

    pup.image_processed = pup.draw_circle_perimeter(image=pup.image_processing.copy(), region=pup.region_pupil)
    pup.image_processed = pup.draw_circle_perimeter(image=pup.image_processed, region=pup.region_iris)
    pup.show_image(pup.image_processed)
    st.pyplot()

    st.table(pup.get_df_results())


def plot_radius_chart():
    pup.radius_pupil_iris.append((pup.region_pupil[2], pup.region_iris[2]))
    st.line_chart(pup.get_chart_data())


def about():
    st.sidebar.title('Sobre')
    st.sidebar.info(
        '''
        Desenvolvido por **[Matheus Bosa](https://bosamatheus.github.io/)** como requisito para avaliação em 
        TE105 Projeto de Graduação do curso de Engenharia Elétrica da Universidade Federal do Paraná (UFPR).
        '''
    )
    st.sidebar.image(image='./resources/profile.png', width=200)

if __name__ == '__main__':
    main()
