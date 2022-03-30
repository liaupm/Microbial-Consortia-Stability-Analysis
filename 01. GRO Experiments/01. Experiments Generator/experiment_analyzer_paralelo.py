def experiment_analyzer_paralelo(experiment_output, n_strains):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    import numpy as np
    import os
    import re
    
   
    
    file = pd.read_csv(experiment_output) #Leemos el archivo csv como un pandas datafram
    headers = file.columns.tolist()
    time = file.time
    file.set_index(headers[0], inplace=True) #El index es el tiempo
    headers = file.columns.tolist()
    file['Total']= file.iloc[:, 0:].sum(axis=1) #Añadimos una columna "Total" donde se guarda la suma del numero de celulas que hay en cada t
    subset = []
    
    #Calculamos la densidad en cada time-step
    
    for strain in range(len(headers)):
        new_col = 'Density' + str(strain + 1)
        file[new_col] = file.iloc[:,strain].div(file.Total)
        subset.append(new_col)
    
    #Calculamos la pendiente de los ultimos 150 time steps con Linear Regression
    t = np.array(time.tolist())
    t = t[-50:]
    
    subset_slopes = []
    last_den = []
    
    for den in range(len(subset)):
        
        d = file[subset[den]]
        d = np.array(d.tolist())
        d = d[-50:]
        
        #Calulamos la media de las densidades de los ultimos 5 time steps para ver si están por encima del threshold,
        #ya que si están muy cerca del cero se considera que la cepa se extingue
        last_den.append(d[-1])
        
        model = linregress(t,d)
        subset_slopes.append(model.slope)
        
    
    cutoff = 0.00025
    
    last_den = np.array(last_den)
    last_den_bool = np.greater(last_den, 0.1)
    stability = []
    
    for j in subset_slopes:
        if abs(j) > cutoff:
            stability.append(1) #UNSTABLE         
        else:
            stability.append(0) #STABLE
    
    
    
    if sum(stability)==0 and last_den_bool.all():
            s = 'STABLE'
            #print(last_den)
    else:
        s = 'UNSTABLE'
    
    
    if n_strains == 2:
        results = {'Slope_Strain1': subset_slopes[0], 'Slope_Strain2': subset_slopes[1], 'Stability': s}
    else:
        results = {'Slope_Strain1': subset_slopes[0], 'Slope_Strain2': subset_slopes[1], 'Slope_Strain3': subset_slopes[2], 'Stability': s}
    
    return(results)