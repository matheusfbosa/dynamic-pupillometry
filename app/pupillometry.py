from image_tools import ImageTools


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
            if self.st.checkbox('Mostrar imagem'):
                self.image_tools.show_image(img)
                self.st.pyplot()

    def pre_processing(self):
        self.st.subheader('Pré-Processamento')

    def pupil_segmentation(self):
        self.st.subheader('Segmentação da pupila')

    def iris_segmentation(self):
        self.st.subheader('Segmentação da íris')

    def results(self):
        self.st.subheader('Resultados')
