import pickle
import os
import pandas as pd
import numpy as np

dir_pickle = 'C:/TFM/ann/Classification/'

data_ann = os.listdir(dir_pickle)

df = pd.DataFrame(columns=['Epochs', 'Learning_Rate', 'Batch_Size', 'Neurons', 'Dropout', 'Activation', 'Initializer', 'Regularizer', 'Optimizer', 'Normalization', 'Acc_train', 'Acc_val', 'Acc_test'])

for data in data_ann:
    with open( dir_pickle + data, 'rb') as handle:
        exp_data = pickle.load(handle)
        
        
        
        epochs =  exp_data['Epochs'][0]
        lr =  exp_data['Learning_Rate'][0]
        batch =  exp_data['Batch_Size'][0]
        neurons =  exp_data['Neurons'][0]
        drop =  exp_data['Dropout'][0]
        act =  exp_data['Activation'][0]
        norm =  exp_data['Normalization'][0]
        init =  exp_data['Initializer'][0]
        reg = exp_data['Regularizer'][0]
        opt = exp_data['Optimizer'][0]
        acc_train = np.mean(exp_data.Acc_train)
        acc_val = np.mean(exp_data.Acc_val)
        acc_test = np.mean(exp_data.Acc_test)
        
        df = df.append({'Epochs': epochs, 'Learning_Rate': lr, 'Batch_Size': batch, 'Neurons': neurons, 'Dropout': drop, 'Activation': act, 'Initializer': init, 'Regularizer': reg, 'Optimizer': opt, 'Normalization': norm, 'Acc_train': acc_train, 'Acc_val': acc_val, 'Acc_test': acc_test}, ignore_index=True)



print('Valor maximo de Acc_train:')
print('\n', df.loc[df['Acc_train'].idxmax()])

print('\n Valor maximo de Acc_val:')
print('\n', df.loc[df['Acc_val'].idxmax()])

print('\n Valor maximo de Acc_test:')
print('\n', df.loc[df['Acc_test'].idxmax()])
        
df.to_csv('C:/TFM/ann/Results Classification/data_ann_results_8.csv')