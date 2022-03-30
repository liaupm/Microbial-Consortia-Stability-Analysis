import os
from keras.models import load_model
import numpy as np
import pandas as pd
from metrics import recall_m, precision_m, f1_m
import os
from gro_writer import gro_writer
from subprocess import call
from experiment_analyzer import experiment_analyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from metrics import recall_m, precision_m, f1_m

path_models = 'C:/TFM/ann/Testing/Models/'
path = 'C:/TFM/ann/Testing/'


##### TEST PSEUDO MODEL ###########

data_rep_red = pd.read_csv(path + 'data_replating_reduced.csv')

type_ann = int(input(
    'Which model do you want to use to make the prediction? \n 1. Classification One-Hot \n 2. Binary \n'))

if type_ann == 1:

    model = load_model(path_models + 'model_classification_onehot_replating.h5')

elif type_ann == 2:

    model = load_model(path_models + 'model_binary_replating.h5', custom_objects={
                   'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})



# Min max the attributes


data_rep_red['Num_Cells_Strain1'] = (
    (data_rep_red.Num_Cells_Strain1 - 1)/(10 - 1)) * (1-(-1)) + (-1)
data_rep_red['Num_Cells_Strain2'] = (
    (data_rep_red.Num_Cells_Strain2 - 1)/(10 - 1)) * (1-(-1)) + (-1)


data_rep_red['t_deg'] = ((data_rep_red.t_deg - 12)/(46 - 12)) * (1-(-1)) + (-1)
data_rep_red['k_degA'] = ((data_rep_red.k_degA - 0.1) /
                          (0.6 - 0.1)) * (1-(-1)) + (-1)
data_rep_red['k_degB'] = ((data_rep_red.k_degB - 0.3) /
                          (0.6 - 0.3)) * (1-(-1)) + (-1)


data_rep_red['ratioS1'] = (
    (data_rep_red.ratioS1 - 0)/(2 - 0)) * (1-(-1)) + (-1)
data_rep_red['ratioS2'] = (
    (data_rep_red.ratioS2 - 0)/(2 - 0)) * (1-(-1)) + (-1)

data_rep_red['Survivors'] = (
    (data_rep_red.Survivors - 0.1)/(0.9 - 0.1)) * (1-(-1)) + (-1)
data_rep_red['Stability'] = (
    (data_rep_red.Stability - 0.0)/(1.0 - 0.0)) * (1-(-1)) + (-1)


labels = data_rep_red.Survivors
att = data_rep_red.drop(columns='Survivors')
att = att.drop(columns=['Unnamed: 0', 'Unnamed: 0.1',
               'Unnamed: 0.1.1', 'posS1', 'posS2'])


df = pd.DataFrame(columns=['Prediction', 'Target'])

survivors = [0.9, 0.7, 0.5, 0.3, 0.1]

for i, row in att.iterrows():

    pred = 0.0

    for survivor in survivors:
        

        num_cells_strain1 = row['Num_Cells_Strain1']
        num_cells_strain2 = row['Num_Cells_Strain2']

        t_deg = row['t_deg']
        k_degA = row['k_degA']
        k_degB = row['k_degB']

        ratioS1 = row['ratioS1']
        ratioS2 = row['ratioS2']

        stability = row['Stability']
        frac_surv = ((survivor - 0.1)/(0.9 - 0.1)) * (1-(-1)) + (-1)

        attributes = pd.DataFrame(columns=['Num_Cells_Strain1', 'Num_Cells_Strain2','t_deg', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'Stability', 'Survivors'])

        attributes = attributes.append({'Num_Cells_Strain1': num_cells_strain1, 'Num_Cells_Strain2': num_cells_strain2, 't_deg': t_deg, 'k_degA': k_degA,
                                       'k_degB': k_degB, 'ratioS1': ratioS1, 'ratioS2': ratioS2, 'Stability': stability, 'Survivors': frac_surv}, ignore_index=True)
        
        prediction = model.predict(attributes)

        if type_ann == 1:

            prediction = np.round(prediction[0]).tolist()

            if prediction == [0.0, 1.0]:
                pred = 1.0
            else:
                pred = 0.0

        elif type_ann == 2:

            pred = np.round(prediction[0])

        if pred == 1.0:
            break
        
        
        
            
    target = labels[i]
    surv = frac_surv

    df = df.append({'Prediction': surv, 'Target': target},
                   ignore_index=True)


df.to_csv(path + 'results_pseudomodel_bin.csv')

# EVALUATE RESULTS OF PSEUDOMODEL



results_bin = pd.read_csv(path + 'results_pseudomodel_bin.csv')
results_onehot = pd.read_csv(path + 'results_pseudomodel_onehot.csv')

results_bin["is_right"] = results_bin["Prediction"]==results_bin["Target"]
results_bin["mse"] = abs((results_bin["Prediction"]-results_bin["Target"]))
results_bin["mse"] = results_bin["mse"] * results_bin["mse"]


mse = results_bin["mse"].sum()/len(results_bin)
accuracy = len(results_bin[results_bin["is_right"]==True])/len(results_bin)

print('Accuracy binary: ', accuracy)
print('MSE binary: ', mse)

results_bin["survivors_pred"] = 0.4 * results_bin["Prediction"] + 0.5
results_bin["survivors_target"] = 0.4 * results_bin["Target"] + 0.5
results_bin["Diff"] = results_bin["survivors_pred"] - results_bin["survivors_target"]

valid = results_bin.query("Diff >= 0")
print("Perturbation predicted bigger that the target:", len(valid)/len(results_bin))


results_onehot["is_right"] = results_onehot["Prediction"]==results_onehot["Target"]
results_onehot["mse"] = abs((results_onehot["Prediction"]-results_onehot["Target"]))
results_onehot["mse"] = results_onehot["mse"] * results_onehot["mse"]


mse = results_onehot["mse"].sum()/len(results_onehot)
accuracy = len(results_onehot[results_onehot["is_right"]==True])/len(results_onehot)

print('Accuracy onehot: ', accuracy)
print('MSE onehot: ', mse)


results_onehot["survivors_pred"] = 0.4 * results_onehot["Prediction"] + 0.5
results_onehot["survivors_target"] = 0.4 * results_onehot["Target"] + 0.5
results_onehot["Diff"] = results_onehot["survivors_pred"] - results_onehot["survivors_target"]

valid = results_onehot.query("Diff >= 0")
print("Perturbation predicted bigger that the target:", len(valid)/len(results_onehot))






