# BIBLIOTECAS

import zipfile
import glob

from datetime import datetime, timedelta

import pandas as pd


# DEFINIR PERÍODO DO DIA PARA ESTUDO

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


# TRATAMENTO CONJUNTO DE DADOS

def treatment_solaredge(files, start_hour, end_hour):
    for name in files:
        zf = zipfile.ZipFile(name)  # Inversor#_Dia
        for spread in zipfile.ZipFile.namelist(zf):
            with zf.open(spread, 'r') as csvfile:
                df = pd.read_csv(csvfile, encoding='utf8', sep=",", engine='python',
                                 on_bad_lines='skip')  # Arquivo CSV - linhas com erro foram desconsideradas

                # Transformar coluna de dia/hora em timestamp
                df['Time'] = pd.to_datetime(df.Time)

                # Criar coluna 'Data' e 'Hora'
                df['Hour'] = [data.time() for data in df['Time']]
                df['Date'] = [data.date() for data in df['Time']]

                # Período do dia
                period_aux = [dt.strftime('%H:%M') for dt in
                              datetime_range(datetime(1, 1, 1, start_hour), datetime(1, 1, 1, end_hour),
                                             timedelta(minutes=5))]
                period_aux = pd.to_datetime(period_aux)  # Tranforma o periodo em timestamp
                period = [data.time() for data in period_aux]  # Transforma timestamp em float
                # print(period)

                # Criar matriz dos dados padronizados
                df_norm = pd.concat([df.loc[:, 'Date'][0:int(len(period))], pd.DataFrame(period)], axis=1)
                df_norm.columns = ['Data', 'Hora']
                for col in df.loc[:, ~df.columns.isin(['Hour', 'Date', 'Time'])]:
                    df_aux = df.loc[:, ['Hour', col]]
                    df_aux = df_aux.dropna()
                    df_aux.reset_index(drop=True, inplace=True)
                    # print(df_aux)
                    avg = []  # Critério de padronização
                    i = 0
                    for time in period:
                        # print(period[i-1], i, time)
                        # print(df_aux[ (df_aux['Hour'] > period[i-1]) & (df_aux['Hour'] <= time) ][col].mean())
                        result = df_aux[(df_aux['Hour'] > period[i - 1]) & (df_aux['Hour'] <= time)][col].mean()
                        avg.append(result)
                        i += 1
                    avg = pd.DataFrame(avg).fillna(0)
                    avg.columns = [col]
                    # print(avg)
                    df_norm = pd.concat([df_norm, avg], axis=1)
                df_norm.to_csv('./DataBase/pre-processing5/{}'.format(spread), mode = 'w')
                print(f'{spread} finalizado')


def treatment_inmet(file, station):
    with zipfile.ZipFile(file) as zf:
        stations = zipfile.ZipFile.namelist(zf)
        csv_vix = [stations[i] for i, value in enumerate([station in name for name in stations]) if value]
        # print(csv_vix)
        for spread in csv_vix:
            with zf.open(spread, 'r') as csvfile:
                df = pd.read_csv(csvfile, delimiter=';', engine='python', skiprows=8, encoding='latin-1')
                df = df.iloc[:, :-1]
                df['Hora'] = df['Hora UTC'].str[:2].astype('timedelta64[h]')  # Mudar o formato da hora - Padronização
                df['Data'] = pd.to_datetime(df.Data)  # Mudar o formato da data - Padronização
        df.to_csv('DataBase/pre-processin/{}'.format(spread), encoding='utf8')


def treatment(files, n_inversor):
    # print(files)
    data_system = {}
    for inversor in range(1, n_inversor + 1):
        print(f'Inversor_{inversor}')
        amp = f'Inversor_{inversor}_Corrente_modulo'
        volt = f'Inversor_{inversor}_Tensao_modulo'
        pot = f'Inversor_{inversor}_Potencia'
        # print(amp)
        # print(volt)
        data_system[f'{amp}'] = [pd.read_csv(files[i], encoding='utf8', sep=",", engine='python', on_bad_lines='skip')
                                 for i, value in enumerate([amp in name for name in files]) if value]
        data_system[f'{volt}'] = [pd.read_csv(files[i], encoding='utf8', sep=",", engine='python', on_bad_lines='skip')
                                  for i, value in enumerate([volt in name for name in files]) if value]
        data_system[f'{pot}'] = [pd.read_csv(files[i], encoding='utf8', sep=",", engine='python', on_bad_lines='skip')
                                 for i, value in enumerate([pot in name for name in files]) if value]
    # Criar função após definido a forma final do dicionário
    '''for key, val in data_system.items():
        print(key)
        for ind in range(4):
            print(ind)
            print(val[ind].describe())
    '''
    # Concatenar linhas dos conjuntos de dados
    data_system_aux = {}
    for key, val in data_system.items():
        data_system_aux[f'{key}'] = pd.concat(val, ignore_index=True)
        aux = pd.concat(val, ignore_index=True)
        aux.to_csv('DataBase/pre-processing/Dataset5/{}.csv'.format(key), encoding='utf8')

    # Concatenar colunas dos conjuntos de dados
    var = ['Corrente_modulo', 'Tensao_modulo', 'Potencia']
    for value in var:
        result = [val for key, val in data_system_aux.items() if value in key]
        result_concat = pd.concat(result, axis=1)
        result_concat = result_concat.loc[:, ~result_concat.columns.duplicated()]
        result_concat = result_concat.loc[:, ~result_concat.columns.isin(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2',
            'Data.1', 'Hora.1', 'Data', 'Hora', 'Data.2', 'Hora.2', 'Data.3', 'Hora.3', 'Inv1 Ica F1 (A)'])]
        result_concat = result_concat.dropna()
        # print(result_concat, '\n', result_concat.columns)
        result_concat.to_csv('DataBase/pre-processing/Dataset5/{}.csv'.format(value))

    return result_concat


# Manipular dicionário
# data[f'{name}_{doc}'] = pd.read_csv(zf.open(spread))

"""
# PROCESSAMENTO SOLAR EDGE

start_hour = 6
end_hour = 20
files = glob.glob("DataBase\DadosInversores-2021\*.zip")                    # Inversor#_Mes-Ano
treatment_solaredge(files, start_hour, end_hour)

# PROCESSAMENTO INMET

file = "DataBase\INMET(31-08-2021).zip"                                     # Dados INMET
station_vix = 'A612'                                                        # Identificação estação Vitória-ES
treatment_inmet(file, station_vix)

# PROCESSAMENTO GERAL

data_files = glob.glob("DataBase\pre-processing\*.csv")
treatment(data_files, n_inversor=4)
"""

start_hour = 6
end_hour = 18
files = glob.glob("Database/DadosInversores-2021/*.zip")
treatment_solaredge(files, start_hour, end_hour)
# data_files = glob.glob("DataBase/pre-processing_5min/*.csv")
# treatment(data_files, n_inversor=4)
