# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:41:45 2021

@author: LIA 2019
"""

import os
import pickle
import pandas as pd

data_experiments = os.listdir('C:/TFM/experimentos_paralelo/data_experiments/') #[:-2]
os.chdir('C:/')

#Without perturbation
df = pd.DataFrame(columns=['Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'posS1', 'posS2', 'Stability'])

#With perturbation
df = pd.DataFrame(columns=['Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'posS1', 'posS2', 'Survivors', 'Stability'])


for data in data_experiments:
    
    with open( 'C:/TFM/experimentos_paralelo/data_experiments/' + data, 'rb') as handle:
        exp_data = pickle.load(handle)
        stability_grade = exp_data['Stability'].value_counts('STABLE')[0]
        num_cells1 =  exp_data['Num_Cells_Strain1'][0]
        num_cells2 =  exp_data['Num_Cells_Strain2'][0]
        t_deg =  exp_data['t_deg'][0]
        k_degA =  exp_data['k_degA'][0]
        k_degB =  exp_data['k_degB'][0]
        ratio_S1 =  exp_data['ratioS1'][0]
        ratio_S2 =  exp_data['ratioS2'][0]
        posS1 = exp_data['posS1'][0]
        posS2 = exp_data['posS2'][0]
        
        df = df.append({'Num_Cells_Strain1': num_cells1, 'Num_Cells_Strain2': num_cells2, 't_deg': t_deg, 'k_degA': k_degA, 'k_degB': k_degB, 'ratioS1': ratio_S1, 'ratioS2': ratio_S2, 'posS1': posS1, 'posS2': posS2, 'Stability': stability_grade }, ignore_index=True)

df.to_csv('C:/TFM/experimentos_paralelo/csv_experiments/data_experiments_2strain_conpos.csv')
    
#creamos un CSV con solo los estables
is_stable = df.loc[df['Stability'].isin([0.9, 1])]
is_stable.to_csv('C:/TFM/experimentos_paralelo/csv_experiments/data_experiments_2strain_stable.csv')
