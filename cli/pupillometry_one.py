import argparse
import os
import csv

from frame_segmentation import segmentation

'''
Estrutura de diretórios:
-> Scripts
-> DATASET
    -> 1
    -> 2
        -> 2_1
        -> 2_2
            -> frame_0000.jpg
            -> frame_0001.jpg
            ...
            -> frame_0149.jpg
'''

def create_results_file(filename):
     with open(filename, mode='w+', newline='\n') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['id_patient', 'row_center', 'col_center', 'radius_pupil', 'radius_iris'])

def write_results(row, col, radius_pupil, radius_iris, filename, pacient):
    with open(filename, mode='a', newline='\n') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([pacient, row, col, radius_pupil, radius_iris])

def pupillometry_one():
    parser = argparse.ArgumentParser(description='Script de segmentação de íris e pupila')
    parser.add_argument('-f', '--folder', default='..', type=str, help='Nome da pasta contendo dataset e pasta com scripts')
    parser.add_argument('-s', '--subfolder', default='1', type=str, help='Nome da subpasta contendo as imagens para segmetação')

    args = parser.parse_args()
    folder = args.folder
    subfolder = args.subfolder

    # Acessa pasta do paciente
    path = os.path.join(folder, 'DATASET', subfolder)
    
    for dirpath, dirnames, filenames in os.walk(path):
        # Filtra pelo tamanho do nome das pastas para não pegar anotações
        dirnames = list(filter(lambda x: len(x) < 5, dirnames))
        
        # Acessa subpasta do paciente
        for dirname in dirnames:
            # Guarda os nomes dos arquivos de imagem
            images = [f for f in os.listdir(os.path.join(path, dirname)) if '.jpg' in f]
            
            if len(images) > 0:
                # Filtra as 150 imagens pegando de 3 em 3, ficando com 50 imagens
                indexes = [i*3 for i in range(0, 50)]
                images = [images[i] for i in indexes]
                                
                for img in images:
                    print(f'Processando frame /{dirname}/{img}...')

                    # Chama script com algoritmo de segmentação da pupila e da íris
                    row, col, radius_pupil, radius_iris = segmentation(os.path.join(path, dirname, img))
                    
                    filename = f'results_{dirname}.csv'
                    
                    # Caso não exista, cria arquivo csv com os resultados
                    if os.path.exists(filename):
                        write_results(row, col, radius_pupil, radius_iris, filename, pacient=dirname)                            
                    # Escreve os resultados no arquivo junto com identificador
                    else:
                        create_results_file(filename)

pupillometry_one()
