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


path_models = 'C:/Users/elenu/Documents/MASTER AI/TFM/Data ANN/Testing/Models/'
path = 'C:/Users/elenu/Documents/MASTER AI/TFM/Data ANN/Testing/'


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


