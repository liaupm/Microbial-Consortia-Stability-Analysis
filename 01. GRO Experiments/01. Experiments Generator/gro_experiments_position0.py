import warnings
warnings.filterwarnings("ignore")
import pickle
import multiprocessing as mp
import time
from gro_writer_paralelo import gro_writer_paralelo
import os
import random
import numpy as np
from shutil import rmtree
import pandas as pd
from subprocess import call
from experiment_analyzer_paralelo import experiment_analyzer_paralelo
from shutil import rmtree
from os import remove





def gro_experiments(gro, rdn_seed, type_exp, circuit, df, n_strains, num_cells, position, k_degA, k_degB, t_deg, frac_surv, ecolis, cont, n):
    
    print('Experimento', cont, ' empezado ')
 
    if type_exp == '2strain':
        data_exp = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 't_deg': t_deg, 'k_degA': k_degA, 'k_degB': k_degB, 'positions': position,  'Replating': frac_surv}
    else:
        data_exp = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 'Num_Cells_Strain3': num_cells[3], 't_deg': t_deg, 'k_degA': k_degA , 'k_degB': k_degB, 'positions': position, 'Replating': frac_surv}

    
    #Calculamos los ratios entre la distancia de las bacterias para ver cuánto de mezcladas están
    posS1 = np.array(data_exp['positions'][:data_exp['Num_Cells_Strain1']])
    posS2 = np.array(data_exp['positions'][data_exp['Num_Cells_Strain1']:data_exp['Num_Cells_Strain1'] + data_exp['Num_Cells_Strain2']])
    distanceS1_S1 = []
    distanceS1_S2 = []
    distanceS2_S2 = []
    distanceS2_S1 = []
      
    for i in range(data_exp['Num_Cells_Strain1']):
        if data_exp['Num_Cells_Strain1']==1:
            distanceS1_S1 = np.append(distanceS1_S1, 0)
            for w in range(data_exp['Num_Cells_Strain2']):
                vect = posS2[w] - posS1[i]
                dist = np.sqrt(sum(vect**2))
                distanceS1_S2 = np.append(distanceS1_S2, dist) #Distancia entre las bacterias del Strain 1 y del Strain 2
        else:
            for j in range(i+1, data_exp['Num_Cells_Strain1']):
                vect = posS1[j] - posS1[i]
                dist = np.sqrt(sum(vect**2))
                distanceS1_S1 = np.append(distanceS1_S1, dist) #Distancia entre las bacterias del Strain 1
            for k in range(data_exp['Num_Cells_Strain2']):
                vect = posS2[k] - posS1[i]
                dist = np.sqrt(sum(vect**2))
                distanceS1_S2 = np.append(distanceS1_S2, dist) #Distancia entre las bacterias del Strain 1 y del Strain 2
        
    for i in range(data_exp['Num_Cells_Strain2']):
        if data_exp['Num_Cells_Strain2']==1:
            distanceS2_S2 = np.append(distanceS2_S2, 0)
        else:
            for j in range(i+1, data_exp['Num_Cells_Strain2']):
                vect = posS2[j] - posS2[i]
                dist = np.sqrt(sum(vect**2))
                distanceS2_S2 = np.append(distanceS2_S2, dist) #Distancia entre las bacterias del Strain 2
             #Distancia entre las bacterias del Strain 1 y del Strain 2 = distance_S1_S2
     
    mean_distance_S1_S1 = np.mean(distanceS1_S1) #Distancia media entre las bacterias del Strain 1
    mean_distance_S2_S2 = np.mean(distanceS2_S2) #Distancia media entre las bacterias del Strain 2
    mean_distance_S1_S2 = np.mean(distanceS1_S2) #Distancia media entre las bacterias del Strain 1 y las del Strain 2
       
    ratio_S1 =  mean_distance_S1_S1 / mean_distance_S1_S2
    ratio_S2 =  mean_distance_S2_S2 /  mean_distance_S1_S2
    
    if n_strains == 3:
        posS3 = np.array(data_exp['positions'][data_exp['Num_Cells_Strain2']:data_exp['Num_Cells_Strain2'] + data_exp['Num_Cells_Strain3']])
        distanceS3_S3 = []
        distanceS3_S1 = []
        distanceS3_S2 = []
        for i in range(data_exp['Num_Cells_Strain3']):
            if data_exp['Num_Cells_Strain3']==1:
                distanceS3_S3 = np.append(distanceS3_S3, 0)
            else:
                for j in range(i+1, data_exp['Num_Cells_Strain3']):
                    vect = posS3[j] - posS3[i]
                    dist = np.sqrt(sum(vect**2))
                    distanceS3_S3 = np.append(distanceS3_S3, dist) #Distancia entre las bacterias del Strain 3
                for k in range(data_exp['Num_Cells_Strain1']):
                    vect = posS1[k] - posS3[i]
                    dist = np.sqrt(sum(vect**2))
                    distanceS3_S1 = np.append(distanceS3_S1, dist) #Distancia entre las bacterias del Strain 1 y del Strain 3
                for k in range(data_exp['Num_Cells_Strain2']):
                    vect = posS2[k] - posS3[i]
                    dist = np.sqrt(sum(vect**2))
                    distanceS3_S2 = np.append(distanceS3_S2, dist) #Distancia entre las bacterias del Strain 2 y del Strain 3
        
        mean_distance_S3_S3 = np.mean(distanceS3_S3) #Distancia media entre las bacterias del Strain 3
        mean_distance_S3_S1 = np.mean(distanceS3_S1) #Distancia media entre las bacterias del Strain 1 y las del Strain 2
        mean_distance_S3_S2 = np.mean(distanceS3_S2) #Distancia media entre las bacterias del Strain 1 y las del Strain 2
          
        ratio_S3_S1 =  mean_distance_S3_S3 / mean_distance_S3_S1
        ratio_S3_S2 =  mean_distance_S3_S3 /  mean_distance_S3_S2
    
    
    #Ejecutamos el experimento con repeticiones
    for repeat in range(10):
        
        rdn_seed = rdn_seed + 3
        
        [direction_exp, direction_exp_output]= gro_writer_paralelo(type_exp, circuit, n_strains, t_deg, k_degA, k_degB, frac_surv, rdn_seed, ecolis, cont, n)
        
        call(gro + direction_exp)
    
        #Analizamos el resultado
        stability_analysis = experiment_analyzer_paralelo(direction_exp_output, n_strains)
        
        # if n_strains ==2:
        #     df = df.append({'Type_Experiment': data_exp['Type_Experiment'], 'Num_Cells_Strain1': data_exp['Num_Cells_Strain1'], 'Num_Cells_Strain2': data_exp['Num_Cells_Strain2'], 't_deg': data_exp['t_deg'], 't_act': data_exp['t_act'], 'k_degA': data_exp['k_degA'], 'k_degB': data_exp['k_degB'], 'ratioS1': ratio_S1, 'ratioS2': ratio_S2, 'Replating': data_exp['Replating'], 'Stability': stability_analysis['Stability'] }, ignore_index=True)
        # else:
        #     df = df.append({'Type_Experiment': data_exp['Type_Experiment'], 'Num_Cells_Strain1': data_exp['Num_Cells_Strain1'], 'Num_Cells_Strain2': data_exp['Num_Cells_Strain2'], 't_deg': data_exp['t_deg'], 't_act': data_exp['t_act'], 'k_degA': data_exp['k_degA'], 'k_degB': data_exp['k_degB'], 'ratioS1': ratio_S1, 'ratioS2': ratio_S2, 'ratioS3_S1': ratio_S3_S1, 'ratioS3_S2': ratio_S3_S2, 'Replating': data_exp['Replating'], 'Stability': stability_analysis['Stability'] }, ignore_index=True)
        
        if n_strains ==2:
            df = df.append({'Type_Experiment': circuit, 'Seed': rdn_seed, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 't_deg': t_deg, 'k_degA': k_degA, 'k_degB': k_degB, 'ratioS1': ratio_S1, 'ratioS2': ratio_S2, 'posS1': posS1, 'posS2': posS2, 'Replating': frac_surv, 'Slope1' : stability_analysis['Slope_Strain1'], 'Slope2' : stability_analysis['Slope_Strain2'], 'Stability': stability_analysis['Stability'] }, ignore_index=True)
        else:
            df = df.append({'Type_Experiment': circuit, 'Seed': rdn_seed, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 'Num_Cells_Strain3': num_cells[2], 't_deg': t_deg, 'k_degA': k_degA, 'k_degB': k_degB, 'ratioS1': ratio_S1, 'ratioS2': ratio_S2, 'ratioS3_S1': ratio_S3_S1, 'ratioS3_S2': ratio_S3_S2, 'posS1': posS1, 'posS2': posS2, 'posS3': posS3,  'Replating': frac_surv, 'Slope1' : stability_analysis['Slope_Strain1'], 'Slope2' : stability_analysis['Slope_Strain2'], 'Slope3' : stability_analysis['Slope_Strain3'] , 'Stability': stability_analysis['Stability'] }, ignore_index=True)
    
        
        
        #with open( 'C:/TFM/experimentos_paralelo/data_experiments/data_' + type_exp + str(cont) + str(repeat) + '.pickle', 'wb') as handle:
        #pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Experimento', cont,' acabado')
    
    with open( 'C:/TFM/experimentos_paralelo/data_experiments/data_' + type_exp + str(cont) + '.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #df.to_csv('C:/TFM/data_experiments/data_' + type_exp + str(cont) + '.csv')
    
   
    #Borramos archivos gro y csv creados para no ocupar memoria
    remove(direction_exp)
    remove(direction_exp_output) 
    
    
    
    
    
    

if __name__ == '__main__':
    

    
    start_time = time.time()
    pool = mp.Pool()
    
    type_exp ='2strain'
    circuit = 'm66' 
    n_strains = 2
    t_degradation = [12, 24, 36, 46]
    k_degradationA = [0.1, 0.3, 0.6]
    k_degradationB = [0.3, 0.5, 0.6]
    replate_frac = [1]
    
    n_pos = 1
    rdn_seed = 1000 #Inicializamos la seed
    
    gro = 'C:/EXE_elegro1-2-3-1_rel/gro/gro.exe '
    
    cont = 0
    n = 0
    
    for pos in range(n_pos):
        
        position = []
        num_cells = []
        ecolis = []
        
        for i in range(n_strains):
            num = random.randint(1,10)
            num_cells.append(num)
            
            for k in range(num):
                #x_position = random.randint(-200,200)
                #y_position = random.randint(-200,200)
                x_position = 0
                y_position = 0
                point = [x_position,y_position]
                position.append(point)
                #position.append(x_position)
                #position.append(y_position)
                #x_position = 0
                #y_position = 0
                ecolis.append( 'ecolis( [ num := 1, x := '+str(x_position)+'.0 , y := '+str(y_position)+'.0 , r := 50 , strain := "Strain'+str(i+1)+'", plasmids := {"p'+str(i+1)+'"}, molecules := {}, mode := "default" ], program p());\n' )

        #Creamos un pandas dataframe para guardar los datos de cada experimento
        if type_exp == '2strain':
            df = pd.DataFrame(columns=['Type_Experiment', 'Seed', 'Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 't_act', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'posS1', 'posS2', 'Replating', 'Slope1', 'Slope2', 'Stability'])
            #'x1', 'y1', 'x2', 'y2'
        elif type_exp == '3strain':
            df = pd.DataFrame(columns=['Type_Experiment', 'Seed', 'Num_Cells_Strain1', 'Num_Cells_Strain2', 'Num_Cells_Strain3', 't_deg', 't_act', 'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'ratioS3_S1', 'ratioS3_S2',  'posS1', 'posS2', 'posS3', 'Slope1', 'Slope2', 'Slope2', 'Replating'])
        else:
            print('Introduce a valid number of strains (2strain/3strain)')
            
         
        for k_degA in k_degradationA:
            
            for k_degB in k_degradationB:
                
                for t_deg in t_degradation:
                    
                    for frac_surv in replate_frac:
                        
                        cont = cont + 1
                        n = n + 1
                        
                        #gro_experiments(gro, rdn_seed, type_exp, circuit, df, n_strains, num_cells, position, k_degA, k_degB, t_deg, t_act, frac_surv, ecolis, cont)
                        
                        pool.apply_async(gro_experiments, args = (gro, rdn_seed, type_exp, circuit, df, n_strains, num_cells, position, k_degA, k_degB, t_deg, frac_surv, ecolis, cont, n)) 
                        
                        rdn_seed = rdn_seed + 10
            
        
    pool.close()
    pool.join()
    print("--- %s seconds ---" % (time.time() - start_time))
