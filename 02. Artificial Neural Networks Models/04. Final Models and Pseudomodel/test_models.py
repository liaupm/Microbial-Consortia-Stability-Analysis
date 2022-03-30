import os
from keras.models import load_model
import numpy as np
import pandas as pd
from metrics import recall_m, precision_m, f1_m
import warnings
warnings.filterwarnings("ignore")
import os
from gro_writer import gro_writer
from subprocess import call
from experiment_analyzer import experiment_analyzer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



    
type_ann = int(input('Which model do you want to use to predict your stability value? \n 1. Classification One-Hot \n 2. Binary \n 3. Regression \n'))


path_models = 'C:/Users/elenu/Documents/MASTER AI/TFM/Data ANN/Testing/Models/'


if type_ann == 1:
    
    model = load_model(path_models + 'model_classification_onehot.h5')
    
elif type_ann == 2:
    
    model = load_model(path_models + 'model_binary.h5', custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})
    
else:
    model = load_model(path_models + 'model_regression.h5', custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})
        

print('Insert the value of the following parameters: \n')
num_cells_strain1 = int(input('Number of bacteria of Strain 1 (min 1 and max 10):'))
num_cells_strain2 = int(input('\n Number of bacteria of Strain 2  (min 1 and max 10):'))
t_deg = int(input('\n Value of the degradation time of bacteriocins (12, 24, 36 or 46 min):'))
k_degA = float(input('\n Value of the degradation constant of QS signals (0.1, 0.3, or 0.6 min^{-1}):'))
k_degB = float(input('\n Value of the degradation constant of bacteriocin signals (0.3, 0.5, or 0.6 min^{-1}):'))


posS1 = np.empty((num_cells_strain1, 2))
posS2 = np.empty((num_cells_strain2, 2))

for i in range(num_cells_strain1):
    
    x = int(input('Value of coordinate x of position of bacteria number ' + str(i+1) +  ' of Strain 1:'))
    y = int(input('Value of coordinate y of position of bacteria numer ' + str(i+1) + ' of Strain 1:'))
    
    posS1[i] = [x, y]

for i in range(num_cells_strain2):
    
    x = int(input('Value of coordinate x of position of bacteria number ' + str(i+1) +  ' of Strain 2:'))
    y = int(input('Value of coordinate y of position of bacteria numer ' + str(i+1) + ' of Strain 2:'))
    
    posS2[i] = [x, y]



#Calculamos los ratios entre la distancia de las bacterias para ver cuánto de mezcladas están

distanceS1_S1 = []
distanceS1_S2 = []
distanceS2_S2 = []
distanceS2_S1 = []
  
for i in range(num_cells_strain1):
    if num_cells_strain1==1:
        distanceS1_S1 = np.append(distanceS1_S1, 0)
        for w in range(num_cells_strain2):
            vect = posS2[w] - posS1[i]
            dist = np.sqrt(sum(vect**2))
            distanceS1_S2 = np.append(distanceS1_S2, dist) #Distancia entre las bacterias del Strain 1 y del Strain 2
    else:
        for j in range(i+1, num_cells_strain1):
            vect = posS1[j] - posS1[i]
            dist = np.sqrt(sum(vect**2))
            distanceS1_S1 = np.append(distanceS1_S1, dist) #Distancia entre las bacterias del Strain 1
        for k in range(num_cells_strain2):
            vect = posS2[k] - posS1[i]
            dist = np.sqrt(sum(vect**2))
            distanceS1_S2 = np.append(distanceS1_S2, dist) #Distancia entre las bacterias del Strain 1 y del Strain 2
    
for i in range(num_cells_strain2):
    if num_cells_strain2==1:
        distanceS2_S2 = np.append(distanceS2_S2, 0)
    else:
        for j in range(i+1, num_cells_strain2):
            vect = posS2[j] - posS2[i]
            dist = np.sqrt(sum(vect**2))
            distanceS2_S2 = np.append(distanceS2_S2, dist) #Distancia entre las bacterias del Strain 2
         #Distancia entre las bacterias del Strain 1 y del Strain 2 = distance_S1_S2
 
mean_distance_S1_S1 = np.mean(distanceS1_S1) #Distancia media entre las bacterias del Strain 1
mean_distance_S2_S2 = np.mean(distanceS2_S2) #Distancia media entre las bacterias del Strain 2
mean_distance_S1_S2 = np.mean(distanceS1_S2) #Distancia media entre las bacterias del Strain 1 y las del Strain 2
   
ratio_S1 =  mean_distance_S1_S1 / mean_distance_S1_S2
ratio_S2 =  mean_distance_S2_S2 /  mean_distance_S1_S2


#Min max los atributos


num_cells_strain1_std = (num_cells_strain1 - 1)/(10 - 1)
num_cells_strain2_std = (num_cells_strain2 - 1)/(10 - 1)

num_cells_strain1_scaled = num_cells_strain1_std * (1-(-1)) + (-1)
num_cells_strain2_scaled = num_cells_strain2_std * (1-(-1)) + (-1)


t_deg_std = (t_deg - 12)/(46 - 12)
k_degA_std = (k_degA - 0.1)/(0.6 - 0.1)
k_degB_std = (k_degB - 0.3)/(0.6 - 0.3)

t_deg_scaled = t_deg_std * (1-(-1)) + (-1)
k_degA_scaled = k_degA_std * (1-(-1)) + (-1)
k_degB_scaled = k_degB_std * (1-(-1)) + (-1)

ratio_S1_std = (ratio_S1 - 0)/(2 - 0)
ratio_S2_std = (ratio_S2 - 0)/(2 - 0)

ratio_S1_scaled = ratio_S1_std  * (1-(-1)) + (-1)
ratio_S2_scaled = ratio_S2_std  * (1-(-1)) + (-1)


att = pd.DataFrame(columns = ['Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2'])

att = att.append({'Num_Cells_Strain1': num_cells_strain1_scaled, 'Num_Cells_Strain2': num_cells_strain2_scaled, 't_deg': t_deg_scaled, 'k_degA': k_degA_scaled, 'k_degB': k_degB_scaled, 'ratioS1': ratio_S1_scaled, 'ratioS2': ratio_S2_scaled}, ignore_index = True)



prediction = model.predict(att)

if type_ann == 1:
    
    prediction = np.round(prediction[0]).tolist()

    if prediction == [0.0, 1.0]:
        print('\n Stable')
    else:
        print('\n Unstable')
    
elif type_ann == 2:
    
    prediction = np.round(prediction[0])

    if prediction == 1.0:
        print('\n Stable')
    else:
        print('\n Unstable')

    
elif type_ann == 3:
    
    prediction = np.round(prediction[0],1)

    print('Stability Score: ', prediction)
    
    
    


type_exp = '2strain'
circuit = 'm62'
n_strains = 2
frac_surv = 1

gro = 'C:/exe_elegro_1-2-3-1_rel/gro/gro.exe '


simulation = input('Do you want to test it in gro? (y/n)\n')

if simulation == 'y':
    
    [direction_exp, direction_exp_output]= gro_writer(type_exp, circuit, n_strains, t_deg, k_degA, k_degB, frac_surv, posS1, posS2)        
    call(gro + direction_exp)
        
    #Analizamos el resultado
    stability_analysis = experiment_analyzer(direction_exp_output, n_strains)
            
    print(stability_analysis)





if type_ann == 3:
    
    action = input('Do you want to know which amount of perturbation stabilizes the consortium? (y/n) \n')
    
    if action == 'y':
        
        type_ann = int(input('Which model do you want to use to make the prediction? \n 1. Classification One-Hot \n 2. Binary \n'))

        if type_ann == 1:
    
            model = load_model(path_models + 'model_classification_onehot_rep.h5')
    
        elif type_ann == 2:
    
            model = load_model(path_models + 'model_binary.h5', custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})
    
        survivors = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        
        
        att_rep = pd.DataFrame(columns = ['Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'stability', 'survivors'])
        
        results = []
        
        
        for i in range(len(survivors)):
            
            att_rep = att_rep.append({'Num_Cells_Strain1': num_cells_strain1_scaled, 'Num_Cells_Strain2': num_cells_strain2_scaled, 't_deg': t_deg_scaled, 'k_degA': k_degA_scaled, 'k_degB': k_degB_scaled, 'ratioS1': ratio_S1_scaled, 'ratioS2': ratio_S2_scaled, 'stability': prediction[0], 'survivors': survivors[i]}, ignore_index = True)

            prediction = model.predict(att_rep)
            
            
            
            if type_ann == 1:
    
                prediction = np.round(prediction[0]).tolist()
            
                if prediction == [0.0, 1.0]:
                    res = 'Stable'
                else:
                    res = 'Unstable'
                
            elif type_ann == 2:
                
                prediction = np.round(prediction[0])
            
                if prediction == 1.0:
                    res = 'Stable'
                else:
                    res = 'Unstable'
            
            results.append(res)
        
            if results[4] == 'Stable':
                 
                 frac_surv = 0.9
        
            elif results[3] == 'Stable':
                
                 frac_surv = 0.7
            
            elif results[2] == 'Stable':
                
                 frac_surv = 0.5
            
            elif results[1] == 'Stable':
                
                 frac_surv = 0.3
            
            elif results[0] == 'Stable':
                
                 frac_surv = 0.1
            
            print('The perturbation that stabilizes the consortium is: ', 1-frac_surv)

        
        simulation = input('Do you want to test it in gro? (y/n)\n')
        
        if simulation == 'y':
    
            


            
            [direction_exp, direction_exp_output]= gro_writer(type_exp, circuit, n_strains, t_deg, k_degA, k_degB, frac_surv, posS1, posS2)        
            call(gro + direction_exp)
                
            #Analizamos el resultado
            stability_analysis = experiment_analyzer(direction_exp_output, n_strains)
                    
            print(stability_analysis)
    
        
       








