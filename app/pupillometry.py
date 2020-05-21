import pandas as pd

from app.image_tools import ImageTools


class Pupillometry:

    def __init__(self, st):
        self.st = st
        self.image_tools = ImageTools()

    def load_image(self):
        self.st.subheader('Carregar imagem')
        img_bytes = self.st.file_uploader(label='Faça o upload da imagem de pupilometria (.jpg ou .png):', 
                                          type=['jpg', 'png'])
        if img_bytes:
            img = self.image_tools.get_image(img_bytes)

            self.st.table(pd.DataFrame(
                index=['Tamanho', 'Valor mínimo', 'Valor máximo'],
                data=[img.shape, img.min(), img.max()],
                columns=['Valor']
            ))

            if self.st.checkbox('Mostrar imagem'):
                self.image_tools.show_image(img)
                self.st.pyplot()

    def pre_processing(self):
        self.st.subheader('Pré-Processamento')

        width = self.st.slider('Tamanho do elemento estruturante')
        selem = self.image_tools.get_selem(width)

        flashes_removed = eq_hist.copy()

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
