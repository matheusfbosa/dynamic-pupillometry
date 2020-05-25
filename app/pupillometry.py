import pandas as pd
import numpy as np

from image_tools import ImageTools


class Pupillometry:

    def __init__(self, st, os):
        self.st = st
        self.os = os
        self.image_in = None
        self.image_pre_processed = None
        self.image_tools = ImageTools(np)

    def load_image(self):
        self.st.subheader('Carregar imagem')

        option = self.st.radio('Selecione:', ('Realizar upload da imagem', 'Utilizar imagem de exemplo'))
        if option == 'Realizar upload da imagem':
            image_bytes = self.st.file_uploader(label='Faça o upload da imagem de pupilometria (.jpg ou .png):', 
                                            type=['jpg', 'png'])
            if image_bytes:
                self.image_in = self.image_tools.get_image(image_bytes)
        elif option == 'Utilizar imagem de exemplo':
            image_bytes = self.os.path.join(self.os.path.dirname(__file__), 'img', 'example.jpg')
            self.image_in = self.image_tools.get_image(image_bytes)

        if self.image_in is not None:
            self.st.subheader('Resumo')
            df = pd.DataFrame(
                index=['Tamanho', 'Valor mínimo', 'Valor máximo'],
                columns=['Valor'],
                data=np.array([self.image_in.shape, self.image_in.min(), self.image_in.max()]),
            )
            self.st.table(df)

            if self.st.checkbox('Mostrar imagem'):
                self.image_tools.show_image(self.image_in)
                self.st.pyplot()
    
    def pre_processing(self):
        self.st.subheader('Pré-Processamento')

        if self.st.checkbox('Filtragem'):
            self.st.text('Filtragem')

            width = self.st.slider('Largura do elemento estruturante quadrado', min_value=1, max_value=20, value=3)
            selem = self.image_tools.get_selem(width)
            self.st.write(selem)
            sigma = self.st.slider('Valor do desvio padrão σ do kernel gaussiano', min_value=.1, max_value=5., value=1.)
            
            if self.st.button('Filtrar'):
                self.__filtering(selem, sigma)
            
        if self.st.checkbox('Remover reflexos na imagem'):
            if self.image_pre_processed:
                self.__remove_flashes(self.image_pre_processed, selem, sigma)
            else:
                self.image_pre_processed = self.image_in.copy()
                self.__remove_flashes(self.image_pre_processed, selem, sigma)

    def __filtering(self, selem, sigma):
        images_plot = self.image_tools.get_filtered_images(self.image_in, selem, sigma)

        self.image_tools.show_all_images(images_plot.values(), 
                                         titles=images_plot.keys())
        self.st.pyplot()

        type_of_filter = self.st.selectbox('Selecione o filtro com melhor resultado:',
                                           ['Média', 'Mediana', 'Gaussiano', 'Equalização de Histograma'])

        self.image_pre_processed = images_plot.get(type_of_filter)
        
    def __remove_flashes(self, image, selem, sigma):
        image_flashes_removed = image.copy()

        L = self.st.slider('Valor do parâmetro L', min_value=1, max_value=50, value=12)
        k = self.st.slider('Valor do parâmetro k', min_value=1.1, max_value=5., value=1.3)

        if self.st.button('Remover Reflexos'):
            self.image_tools.remove_flashes(image_flashes_removed, L, k)
            image_flashes_removed_blurred = self.image_tools.filter_image('Mediana', image_flashes_removed, selem, sigma)

            self.image_tools.show_all_images([image, image_flashes_removed, image_flashes_removed_blurred], 
                                         titles=['Antes', 'Depois', 'Suavizada'])

    def pupil_segmentation(self):
        self.st.subheader('Segmentação da pupila')

    def iris_segmentation(self):
        self.st.subheader('Segmentação da íris')

    def results(self):
        self.st.subheader('Resultados')
