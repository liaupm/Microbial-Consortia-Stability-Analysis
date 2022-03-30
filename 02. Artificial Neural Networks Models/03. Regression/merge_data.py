import pickle
import os
import pandas as pd
import numpy as np



#Experimentos 3 posiciones
dir_results = 'C:/TFM/ann/Regression Results/'
results_file = os.listdir(dir_results)




df = pd.DataFrame()

for csv in results_file:
    
    data = pd.read_csv(dir_results + csv)
    df = pd.concat([df, data], axis=0).reset_index(drop=True)
    

df = df.drop(columns = ['Unnamed: 0'])
df = df.dropna(subset=['Acc_test'])
df = df.fillna(value = {'Optimizer': 'adam'})



df.to_csv('C:/TFM/ann/data_ann_results_regression.csv')