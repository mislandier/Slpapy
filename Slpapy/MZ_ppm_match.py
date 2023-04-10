import numpy as np
import pandas as pd


def Match_mz_value(lib, data, ppm):
    lib = str(lib)
    data = str(data)
    lib = pd.read_csv(f'{lib}', header=0, low_memory=False)
    data = pd.read_csv(f'{data}', header=0, low_memory=False)
    list = pd.DataFrame()
    list['lib'] = lib
    list['up'] = (ppm / 1000000) * list['lib'] + list['lib']
    list['low'] = -((ppm / 1000000) * list['lib'] - list['lib'])
    list['m/z'] = np.nan
    for i in range(len(lib)):
        for j in range(len(data)):
            if list.loc[i, 'up'] > data[j] > list.loc[i, 'low']:
                if [abs(list.loc[i, 'lib'] - list.loc[i, 'm/z']) > abs(list.loc[i, 'lib'] - data[j])] or list.loc[
                    i, 'm/z'] == np.nan:
                    list.loc[i, 'number'] = j + 3
                    list.loc[i, 'm/z'] = data[j]
                    list.loc[i, 'error_ppm'] = (abs(list.loc[i, 'lib'] - data[j]) / list.loc[i, 'lib']) * 1000000
    list.to_csv('Match_mz_value.csv')
    return list
