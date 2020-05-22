import pandas as pd
import numpy as np

from image_tools import ImageTools


class Pupillometry:

    def __init__(self, st, os):
        self.st = st
        self.os = os
        self.image_in = None
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
            
            images_plot = self.image_tools.filtering(self.image_in, selem, sigma)
            self.image_tools.show_all_images(images_plot, 
                                             titles=['Média', 'Mediana', 'Gaussiano', 'Equalização de Histograma'])
            self.st.pyplot()

            type_of_filter = self.st.selectbox('Selecione o filtro com melhor resultado:',
                                               ['Média', 'Mediana', 'Gaussiano', 'Equalização de Histograma'])

        if self.st.checkbox('Remover reflexos na imagem'):
            __remove_flashes(img)    

    def __filtering(self, selem):
        image_show_all(original, mean, median, gaussian, 
                    titles=['Original', 'Média', 'Mediana', 'Gaussiano']);
        
        type_of_filter = st.selectbox('Selecione o tipo de filtro:', ['Média', 'Mediana', 'Gaussiano', 'Equalização de Histograma'])
        
    def __remove_flashes(self, img):
        flashes_removed = img.copy()

        # L e k definidos empiricamente
        L = 12
        k = 1.3
        self.image_tools.remove_flashes(flashes_removed, L, k)
        flashes_removed_final = median_filter(img_as_ubyte(flashes_removed), square(L))

        image_show_all(original, flashes_removed, flashes_removed_final, 
                    titles=['Original', 'Processada', 'Processada e filtrada']);

    def pupil_segmentation(self):
        self.st.subheader('Segmentação da pupila')

    def iris_segmentation(self):
        self.st.subheader('Segmentação da íris')

    def results(self):
        self.st.subheader('Resultados')
