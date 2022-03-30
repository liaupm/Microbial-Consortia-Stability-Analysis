#Función que crea los experimentos de 2 strains y de 3 strains en función de los parámetros dados


def gro_writer_paralelo(type_exp, circuit, n_strains, deg, k_degA, k_degB, frac_surv, rdn_seed, ecolis, cont):
    
    import random
    import pandas as pd

    
    ## INICIO PROGRAMA GRO ##
    ## (común para todos los experimentos) ##
    
    start = open ('initial_parameters.GRO', 'r')                             
    initial = start.readlines()
    start.close()
    
    ## MODELO A SIMULAR ##
    model = open ('C:/TFM/experimentos_paralelo/' + type_exp + '/' + circuit + '.GRO', 'r')                               
    model_type = model.readlines()
    model.close()

    ## FIN PROGRAMA GRO ##
    ## (común para todos los experimentos) ##
    stop = open ('end_parameters.GRO', 'r')                               
    end = stop.readlines()
    stop.close()

    ###### CREACION ARCHIVO .GRO #########
     
    name_experiment= circuit + '_output' + str(cont) + '.GRO'  
    file = open('C:/TFM/experimentos_paralelo/' + name_experiment, 'w')  


    ####### ESCRIBIR EN EL ARCHIVO GENERADO #################
    file.writelines(initial)
    
    seed ='set ("seed",' + str(rdn_seed) + '); \n'
    file.write(seed)
    
    
    file.write('//------------------------------------------------------- \n')
    file.write('//################################################# \n')
    file.write('//############# GLOBAL VARIABLES  ############## \n')
    file.write('//################################################# \n')

    t_deg ='t_deg := ' + str(deg) + '; \n'
    file.write(t_deg)
    

    kdegB ='k_degB := ' + str(k_degB) + '; \n'
    file.write(kdegB)
    
    kdegA ='k_degA := ' + str(k_degA) + '; \n'
    file.write(kdegA)
    
    name_output_file = 'output_file_name := "' + circuit + '_' + str(cont) + '_output";\n'
    file.write(name_output_file)
    
    name_output_path = 'output_path_name := "C:/TFM/experimentos_paralelo/output_gro/"; \n'
    file.write(name_output_path)
    
    file.writelines(model_type)
    
    if frac_surv != 1:
        
        #Si hay replating, añadir la función de replating
        
        file.write('//------------------------------------------------------- \n')
        file.write('//################################################# \n')
        file.write('//#############  REPLATING FUNCTION  ############## \n')
        file.write('//################################################# \n')
        
        survivors_replating = 'survivors := ' + str(frac_surv) + '; \n';
        file.write(survivors_replating)
        
        replate_timing = 'replate_timing := [ output_time_start := 500.0, output_time_stop := 10000.0, period := -1 ]; \n'
        file.write(replate_timing)
        
        
        survivors_variability = 'survivors_variability := 0.0; \n';
        file.write(survivors_variability)
        
        replating = 'replating([ alive_fraction := survivors, alive_fraction_var := survivors_variability, timing := replate_timing ]);\n'
        file.write(replating)

    
    file.writelines(end)
    
    for bacteria in ecolis:
        file.write(bacteria)

   
    # position = []
    # num_cells = []
    
    # for i in range(n_strains):
        
    #     num = random.randint(1,5)
    #     num_cells.append(num)
        
    #     for k in range(num):
    #         x_position = random.randint(-200,200)
    #         y_position = random.randint(-200,200)
    #         point = [x_position,y_position]
    #         position.append(point)
    #         #position.append(x_position)
    #         #position.append(y_position)
    #         #x_position = 0
    #         #y_position = 0
    #         ecolis= 'ecolis( [ num := 1, x := '+str(x_position)+'.0 , y := '+str(y_position)+'.0 , r := 50 , strain := "Strain'+str(i+1)+'", plasmids := {"p'+str(i+1)+'"}, molecules := {}, mode := "default" ], program p());\n'
    #         file.write(ecolis)
        
    
    file.close()
    
    # if type_exp == '2strain':
    #     #data = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 't_deg': deg, 't_act': act, 'k_deg': k_deg, 'x1_1': position[0], 'y1_1': position[1], 'x2_1': position[2], 'y2_1': position[3], 'x3_1': position[4], 'y3_1': position[5], 'x1_2': position[6], 'y1_2': position[7], 'x2_2': position[8], 'y2_2': position[9], 'x3_2': position[10], 'y3_2': position[11],  'Replating': frac_surv}
    #     data_exp = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[1], 't_deg': deg, 't_act': act, 'k_degA': k_degA, 'k_degB': k_degB, 'positions': position,  'Replating': frac_surv}
    # else:
    #     data_exp = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[2], 't_deg': deg, 't_act': act, 'k_degA': k_degA , 'k_degB': k_degB, 'positions': position, 'Replating': frac_surv}
    #     #data = {'Type_Experiment': circuit, 'Num_Cells_Strain1': num_cells[0], 'Num_Cells_Strain2': num_cells[2], 't_deg': deg, 't_act': act, 'k_deg': k_deg , 'x1_1': position[0], 'y1_1': position[1], 'x2_1': position[2], 'y2_1': position[3], 'x3_1': position[4], 'y3_1': position[5], 'x1_2': position[6], 'y1_2': position[7], 'x2_2': position[8], 'y2_2': position[9], 'x3_2': position[10], 'y3_2': position[11], 'Replating': frac_surv}

    
    direction_exp = 'C:/TFM/experimentos_paralelo/' + name_experiment
    direction_exp_output = 'C:/TFM/experimentos_paralelo/output_gro/' + circuit + '_' + str(cont) + '_output.csv'

    return direction_exp, direction_exp_output
