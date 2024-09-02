#cleans the original dataset from Sao Paulo's Portal da Transparencia
import pandas as pd
import os
import numpy as np
import sys

def clean(srcFile, destFile, delimiter):
    df = 0
    # if srcFile[-3:] == 'csv':
    df = pd.read_csv(srcFile, delimiter=delimiter)
    # elif srcFile[-4:] == 'xlsx':
    #     df = pd.read_excel(srcFile, delimiter=delimiter)
    #select only those of cargo theft
    df = df[df.DESCR_CONDUTA == 'Carga']
    df = df[df.RUBRICA == 'Roubo (art. 157)']

    columnsToDrop = [
        'ID_DELEGACIA', 'NOME_DEPARTAMENTO', 'NOME_SECCIONAL', 'NOME_DELEGACIA',
        'NOME_MUNICIPIO', 'ANO_BO', 'NUM_BO',
        'NOME_DEPARTAMENTO_CIRC', 'NOME_SECCIONAL_CIRC', 'NOME_DELEGACIA_CIRC',
        'NOME_MUNICIPIO_CIRC', 'AUTORIA_BO',
        'FLAG_INTOLERANCIA', 'TIPO_INTOLERANCIA', 'FLAG_FLAGRANTE',
        'FLAG_STATUS', 'DESC_LEI', 'FLAG_ATO_INFRACIONAL',
        'DESCR_CONDUTA', 'DESDOBRAMENTO', 'CIRCUNSTANCIA', 'DESCR_TIPOLOCAL',
        'CONT_VEICULO', 'DESCR_PERIODO', 'CIDADE.1', 'CEP',
        'DESCR_MARCA_VEICULO', 'DESCRICAO_APRESENTACAO', 'DATAHORA_REGISTRO_BO', 'DATA_COMUNICACAO_BO', 'DATAHORA_IMPRESSAO_BO',
        'ANO_FABRICACAO', 'ANO_MODELO', 'PLACA_VEICULO', 'DESC_COR_VEICULO',
        'LOGRADOURO_VERSAO', 'LOGRADOURO', 'NUMERO_LOGRADOURO', 'VERSAO']

    df.drop(columnsToDrop, axis=1, inplace=True)
    df = df.drop_duplicates()

    #converts lat long to float64
    df[['LATITUDE', 'LONGITUDE']] = df[['LATITUDE', 'LONGITUDE']].apply(lambda x : x.str.replace(",", "."), axis=1)
    df['LATITUDE'] = df['LATITUDE'].astype(np.float64)
   
    df['LONGITUDE'] = df['LONGITUDE'].astype(np.float64)
    df.to_csv(destFile, index=False)  

if len(sys.argv) == 1:
    print("Usage: cleaner.py file1 file2 ... fileN")
    print("Writes output to data_processed folder")
    sys.exit()
for i in range(1, len(sys.argv)):
    clean(sys.argv[i], "data_processed/" + os.path.basename(sys.argv[i]) + "_cleaned.csv", ";")