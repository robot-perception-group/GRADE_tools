import os
import time
from datetime import date
import socket
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mapping_grade = {"75bf6":"D",
                "b13a4":"DH",
                "53bfe":"F",
                "d94ec":"FH",
                "12e46":"S",
                "b0a9c":"SH",
                "23aae":"WO",
                "d8c14":"WOH"}

if __name__ == '__main__':

    RES_PATHS = ['/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaSLAM',
                '/ps/project/irotate/GRADE_new_res/networks_variations/GRADE/DynaVINS']
    # RES_PATHS = ['/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaSLAM',
    #             '/ps/project/irotate/GRADE_new_res/networks_variations/TUM/DynaVINS']
    # lengths = {
    # 'halfsphere': 36.08,
    # 'rpy': 30.94,
    # 'xyz':29.06,
    # 'static':25.15}
    # create dictionary
    res_dict = {'DynaVINS':{}, 'DynaSLAM':{}}
    lengths = {'D':60., 'DH':60., 'F':60., 'FH':60., 'S':60., 'SH':60., 'WO':60., 'WOH':60.}

    for path in RES_PATHS:
        for network in os.listdir(path):
            type_path = os.path.join(path, network)
            for id_folder in os.listdir(type_path):
                print('processing_path:', path)
                id_folder_mapped = ''
                for key in mapping_grade.keys():
                    if key in id_folder:
                        id_folder_mapped = mapping_grade[key]
                        break
                if id_folder_mapped == '':
                    id_folder_mapped = id_folder

                if id_folder_mapped not in res_dict[path.split('/')[-1]]:
                    res_dict[path.split('/')[-1]][id_folder_mapped] = {'ate':[], 'ate_std':[], 'mt':[], 'mt_std':[], 'keys':[]}
                with open(os.path.join(type_path, id_folder, 'new_res.txt'), 'r') as f:
                    try:
                        c = f.readlines()
                        ate = float(c[0].split(' ')[2])
                        ate_std = float(c[0].split(' ')[4])
                        mt = float(c[1].split(' ')[2]) # this is for missing time
                        mt_std = float(c[1].split(' ')[4])
                        tr = c[2].split('[')[1].split(']')[0].split(' ')
                        sequence_length = lengths[id_folder_mapped]
                        tr = [(sequence_length - float(x))/sequence_length for x in tr if x != '']

                        res_dict[path.split('/')[-1]][id_folder_mapped]['ate'].append(ate)
                        res_dict[path.split('/')[-1]][id_folder_mapped]['ate_std'].append(ate_std)
                        # res_dict[path.split('/')[-1]][id_folder_mapped]['mt'].append(mt)
                        # res_dict[path.split('/')[-1]][id_folder_mapped]['mt_std'].append(mt_std)
                        res_dict[path.split('/')[-1]][id_folder_mapped]['tr'].append(np.round(np.mean(tr), 3))
                        res_dict[path.split('/')[-1]][id_folder_mapped]['tr_std'].append(np.round(np.std(tr), 3))
                        res_dict[path.split('/')[-1]][id_folder_mapped]['keys'].append(network)
                    except Exception as e:
                        print(e)

    # Convert nested dictionary to DataFrame
    for network in res_dict.keys():
        print(network)
        data = res_dict[network]
        dfs = []
        for key, values in data.items():
            df = pd.DataFrame(values)
            df['Type'] = key
            dfs.append(df)

        # Concatenate all DataFrames
        result_df = pd.concat(dfs, ignore_index=True)

        # Melt the table
        result_df = result_df.melt(id_vars=['Type', 'keys'], var_name='Metric', value_name='Value')

        # Create rows for average and std
        result_df['Metric_Type'] = result_df['Metric'].apply(lambda x: 'average' if 'std' not in x else 'std')
        result_df['Metric'] = result_df['Metric'].str.replace('_std', '')

        # Pivot the table
        result_df = result_df.pivot_table(index=['Type', 'Metric_Type'], columns=['Metric', 'keys'], values='Value')

        # Flatten the column MultiIndex
        result_df.columns = [f'{metric}_{key}' for metric, key in result_df.columns]

        # Order the columns
        ordered_columns = []
        for key in data[list(data.keys())[0]]['keys']:
          ordered_columns.extend([f'ate_{key}', f'mt_{key}'])

        result_df = result_df[ordered_columns]

        # Reset the index to make 'Type' and 'Metric_Type' columns again
        result_df = result_df.reset_index()

        # Display the final DataFrame
        print(result_df.to_latex(float_format="%.3f"))