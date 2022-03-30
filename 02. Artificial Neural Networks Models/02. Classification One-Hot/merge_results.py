import pickle
import os
import pandas as pd
import numpy as np



#Experimentos 3 posiciones
dir_results = 'C:/TFM/ann/Results Classification/'
results_file = os.listdir(dir_results)




df = pd.DataFrame()

for csv in results_file:
    
    data = pd.read_csv(dir_results + csv)
    df = pd.concat([df, data], axis=0).reset_index(drop=True)
    

df = df.drop(columns = ['Unnamed: 0'])
df = df.dropna(subset=['Acc_test'])
df = df.fillna(value = {'Optimizer': 'adam'})



df.to_csv('C:/TFM/ann/data_ann_results_classification_onehot.csv')


print('Valor maximo de Acc_train:')
print('\n', df.loc[df['Acc_train'].idxmax()])

print('\n Valor maximo de Acc_val:')
print('\n', df.loc[df['Acc_val'].idxmax()])

print('\n Valor maximo de Acc_test:')
print('\n', df.loc[df['Acc_test'].idxmax()])