import streamlit as st
import os
import pandas as pd

from pupillometry import Pupillometry



@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_pupillometry():
    return Pupillometry()


pup = get_pupillometry()


def main():
    with open(os.path.join(os.path.dirname(__file__), 'styles.css')) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.sidebar.title('Pupilometria Dinâmica')
    st.sidebar.info('Aplicação que realiza pupilometria dinâmica em imagens.')

    pup = get_pupillometry()

    if st.sidebar.checkbox('Carregar imagem'):
        load_image()
    
    if st.sidebar.checkbox('Pré-Processamento'):
        pre_processing()
    
    if st.sidebar.checkbox('Segmentação da pupila'):
        pupil_segmentation()

    if st.sidebar.checkbox('Segmentação da íris'):
        iris_segmentation()

    if st.sidebar.checkbox('Resultados'):
        results()

    about()


def load_image():
    st.subheader('Carregar imagem')

    option_image = st.radio('Selecione:', ('Realizar upload da imagem', 'Utilizar imagem de exemplo'))

    if option_image == 'Realizar upload da imagem':
        image_bytes = st.file_uploader(label='Faça o upload da imagem de pupilometria (.jpg ou .png):', 
                                       type=['jpg', 'png'])
        if image_bytes:
            pup.get_image(image_bytes)

    elif option_image == 'Utilizar imagem de exemplo':
        image_bytes = os.path.join(os.path.dirname(__file__), 'images', 'example.jpg')
        pup.get_image(image_bytes)

    if pup.image_in is not None:
        st.subheader('Resumo')
        df = pd.DataFrame(
            index=['Tamanho', 'Valor mínimo', 'Valor máximo'],
            columns=['Valor'],
            data=pup.get_np_array([pup.image_in.shape, pup.image_in.min(), pup.image_in.max()])
        )
        st.table(df)

        if st.checkbox('Mostrar imagem'):
            pup.show_image(pup.image_in)
            st.pyplot()


def pre_processing():
    st.subheader('Pré-Processamento')
        
    if st.checkbox('Filtragem'):

        width = st.slider('Largura do elemento estruturante quadrado:', min_value=1, max_value=20, value=3)
        selem = pup.get_selem(width)
        st.write(selem)
        sigma = st.slider('Valor do desvio padrão σ do kernel gaussiano:', min_value=.1, max_value=5., value=1.)
        
        images_to_plot = __filtering(selem, sigma)
        pup.show_all_images(images_to_plot.values(), titles=pup.type_of_filters, nrows=2, ncols=2)
        st.pyplot()

        type_of_filter_selected = st.selectbox('Selecione o filtro com melhor resultado:',
                                    pup.type_of_filters)
        pup.image_filtered = images_to_plot.get(type_of_filter_selected).copy()

    if st.checkbox('Remover reflexos na imagem'):

        # Mostrar imagem especificando o que são os parâmetros L e k
        L = st.slider('Valor do parâmetro L:', min_value=1, max_value=50, value=12)
        k = st.slider('Valor do parâmetro k:', min_value=1.1, max_value=5.0, value=1.3)
        width = st.slider('Largura do elemento estruturante quadrado para filtragem de suavização:', min_value=1, max_value=20, value=3)
        selem = pup.get_selem(width)
        st.write(selem)

        st.info('Removendo reflexos...')
        artifacts = __remove_flashes(L, k, selem)

        if len(artifacts) > 0:
            st.success('Reflexos removidos em: ' + str(artifacts).strip('[]') + '.')
            pup.show_all_images([pup.image_filtered, pup.image_pre_processed], 
                                titles=['Antes', 'Depois'], nrows=1, ncols=2)
            st.pyplot()
        else:
            st.success('Nenhum reflexo foi encontrado a partir dos parâmetros.')


@st.cache(suppress_st_warning=True)
def __filtering(selem, sigma):
    return pup.get_filtered_images(selem, sigma, image=pup.image_in)


@st.cache(suppress_st_warning=True)
def __remove_flashes(L, k, selem):
    image = pup.image_filtered.copy()
    artifacts = pup.remove_flashes(L, k, image)
    pup.image_pre_processed = pup.filter_image(image, selem=selem, sigma=None, type_of_filter='Mediana')
    return artifacts


def pupil_segmentation():
    st.subheader('Segmentação da pupila')

    if st.checkbox('Filtro de Canny e transformada circular de Hough'):

        st.markdown('Filtro de Canny')
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
        image_edges = __canny_filter(sigma, low_threshold, high_threshold)
        st.success('Bordas detectadas.')
        pup.show_image(image_edges)
        st.pyplot()

        st.markdown('Transformada circular de Hough')
        st.info('Buscando por circunferências...')
        pup.region_pupil = __circle_hough_transform(image_edges)

        if pup.region_pupil is not None:
            st.success('Pupila encontrada em ('
                       + str(pup.region_pupil[0]) + ', ' 
                       + str(pup.region_pupil[1]) + ') com raio de '
                       + str(pup.region_pupil[2]) + ' pixels.')

            pup.image_pupil = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_pupil)

            pup.show_image(pup.image_pupil)
            st.pyplot()

        else:
            st.warning('A região da pupila não foi encontrada na imagem.')
            pup.image_pupil = None



@st.cache(suppress_st_warning=True)
def __canny_filter(sigma, low_threshold, high_threshold):
    return pup.get_edges_canny(sigma, low_threshold, high_threshold, image=pup.image_pre_processed.copy())


@st.cache(suppress_st_warning=True)
def __circle_hough_transform(image_edges):
    return pup.pupil_segmentation_hough(image_edges.copy())


def iris_segmentation():
    st.subheader('Segmentação da íris')

    st.info('Buscando pela região de fronteira entre íris e esclera..')
    pup.region_iris = __iris_segmentation_horizontal_gradient()

    if pup.region_iris is not None:
        st.success('Íris encontrada em ('
                    + str(pup.region_iris[0]) + ', ' 
                    + str(pup.region_iris[1]) + ') com raio de '
                    + str(pup.region_iris[2]) + ' pixels.')

        pup.image_iris = pup.draw_circle_perimeter(image=pup.image_pre_processed.copy(), region=pup.region_iris)

        pup.show_image(pup.image_iris)
        st.pyplot()

    else:
        st.warning('A região da íris não foi encontrada na imagem.')
        pup.image_iris = None
   

def __iris_segmentation_horizontal_gradient():
    return pup.iris_segmentation_horizontal_gradient(image=pup.image_pre_processed.copy())


def results():
    st.subheader('Resultados')

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
    #image_profile_path = os.path.join(os.path.dirname(__file__), 'images', 'profile.jpg')
    image_profile_path = './images/profile.jpg'
    st.sidebar.image(image_profile_path, use_column_width=True)
    st.sidebar.info(
        '''
        Desenvolvido por **Matheus Bosa** como requisito para avaliação em TE105 Projeto de Graduação
        do curso de Engenharia Elétrica da Universidade Federal do Paraná (UFPR).
        [LinkedIn](https://www.linkedin.com/in/matheusbosa/) e [GitHub](https://github.com/bosamatheus).
        '''
    )


if __name__ == '__main__':
    main()
