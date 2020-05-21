import os
import streamlit as st

from app.pupillometry import Pupillometry


def main():
    st.sidebar.image(os.path.join(os.path.dirname(__file__), 'img', 'profile.jpg'), use_column_width=True)

    with open(os.path.join(os.path.dirname(__file__), 'styles', 'style.css')) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.sidebar.title('aaPupilometria Dinâmica')    
    st.sidebar.info('Aplicação que realiza pupilometria dinâmica.')

    pupillometry = Pupillometry(st)

    if st.sidebar.checkbox('Carregar imagem'):
        pupillometry.load_image()
    
    if st.sidebar.checkbox('Pré-Processamento'):
        pupillometry.pre_processing()
    
    if st.sidebar.checkbox('Segmentação da pupila'):
        pupillometry.pupil_segmentation()

    if st.sidebar.checkbox('Segmentação da íris'):
        pupillometry.iris_segmentation()

    if st.sidebar.checkbox('Resultados'):
        pupillometry.results()

    about()

def about():
    st.sidebar.title('Sobre')    
    st.sidebar.image(os.path.join(os.path.dirname(__file__), 'img', 'profile.jpg'), use_column_width=True)
    st.sidebar.info(
        '''
        Desenvolvido por **Matheus Bosa** como requisito para avaliação em TE105 Projeto de Graduação
        do curso de Engenharia Elétrica da Universidade Federal do Paraná (UFPR).
        [LinkedIn](https://www.linkedin.com/in/matheusbosa/) e [GitHub](https://github.com/bosamatheus).
        '''
    )

if __name__ == '__main__':
    main()
