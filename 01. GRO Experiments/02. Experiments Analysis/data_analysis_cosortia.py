import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import re



mpl.rc('font',family='Times New Roman')
titfont = {'fontname': 'Times New Roman', 'weight': 'bold', 'fontsize':'15'}
axsfont = {'fontname': 'Times New Roman', 'fontsize':'12'}



num = '4125'
circuit = 'm' + num
circuit_tit = '$m_{' + num + '}$'
nstrain = 3


path_fig = 'C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/' + circuit

if num == '48':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m48_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m48_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m48_100pos.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m48_100pos_replating.csv')
elif num == '66':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m66_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m66_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m66_100pos.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m66_100pos_replating.csv')
elif num == '62':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m62_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m62_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m62_100pos.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m62_100pos_replating.csv')
elif num == '3938':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m3938_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m3938_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m3938_100pos0305.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m3938_100pos_replating_0305.csv')
elif num == '4119':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4119_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4119_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4119_100pos0305.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4119_100pos_replating_0305.csv')
elif num == '4125':
    df4 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4125_4pos.csv')
    df4rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4125_4pos_rep.csv')
    df100 = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4125_100pos.csv')
    df100rep = pd.read_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Creador de experimentos/Experimentos Consorcios/m4125_100pos_replating.csv')

    



######################        4 POSICIONES             ##############################


#Contar de los 144, cuantos son estables y cuantos inestables
index, counts = np.unique(df4[['Stability']].to_numpy(), return_counts=True)
suma_stability = pd.Series(counts,  index)


#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y T_DEG #
a = df4.groupby(by = ['Stability', 't_deg']).size()
a = a.reset_index()

s = a['Stability'].unique().tolist()
t = a['t_deg'].unique().tolist()
st = pd.DataFrame(  index= s, columns= t)

line = []
i = 0

for stab in s:
    for tdeg in t:
        d = a[a.Stability.isin([stab])&a.t_deg.isin([tdeg])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    st.iloc[i] = line
    line  = []
    i = i + 1


st.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + ' - $t_{deg}$')
plt.yticks(range(0, 25, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $t_{deg}$',  **titfont)

plt.savefig(path_fig + 'stability_tdeg_' + circuit + '.png')




#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y K_DEGA #
b = df4.groupby(by = ['Stability', 'k_degA']).size()
b = b.reset_index()

s = b['Stability'].unique().tolist()
kA = b['k_degA'].unique().tolist()
skA = pd.DataFrame(  index= s, columns= kA)

line = []
i = 0

for stab in s:
    for kdegA in kA:
        d = b[b.Stability.isin([stab])&b.k_degA.isin([kdegA])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    skA.iloc[i] = line
    line  = []
    i = i + 1

skA.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + ' - $k_{degA}$')
plt.yticks(range(0, 35, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $k_{degA}$',  **titfont)
plt.savefig(path_fig + 'stability_kA_' + circuit + '.png')


#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y K_DEGB #
c = df4.groupby(by = ['Stability', 'k_degB']).size()
c = c.reset_index()

s = c['Stability'].unique().tolist()
kB = c['k_degB'].unique().tolist()
skB = pd.DataFrame(  index= s, columns= kB)

line = []
i = 0

for stab in s:
    for kdegB in kB:
        d = c[c.Stability.isin([stab])&c.k_degB.isin([kdegB])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    skB.iloc[i] = line
    line  = []
    i = i + 1

skB.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + '- $k_{degB}$')
plt.yticks(range(0, 40, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $k_{degB}$',  **titfont)
plt.savefig(path_fig + 'stability_kB_' + circuit + '.png')








###################                  4 POSICIONES REPLATING            #############


#Contar de estables, cuantos son estables y cuantos inestables con replating
indexrep, countsrep = np.unique(df4rep[['New_Stability']].to_numpy(), return_counts=True)
suma_stability_rep = pd.Series(countsrep,  indexrep)


#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y T_DEG #
a = df4rep.groupby(by = ['New_Stability', 't_deg']).size()
a = a.reset_index()

s = a['New_Stability'].unique().tolist()
t = a['t_deg'].unique().tolist()
st = pd.DataFrame(  index= s, columns= t)

line = []
i = 0

for stab in s:
    for tdeg in t:
        d = a[a.New_Stability.isin([stab])&a.t_deg.isin([tdeg])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    st.iloc[i] = line
    line  = []
    i = i + 1


st.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + '$ - t_{deg}$')
plt.yticks(range(0, 15, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $t_{deg}$ - Replating',  **titfont)

plt.savefig(path_fig + 'stability_tdeg_rep_' + circuit + '.png')



#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y K_DEGA #
b = df4rep.groupby(by = ['New_Stability', 'k_degA']).size()
b = b.reset_index()

s = b['New_Stability'].unique().tolist()
kA = b['k_degA'].unique().tolist()
skA = pd.DataFrame(  index= s, columns= kA)

line = []
i = 0

for stab in s:
    for kdegA in kA:
        d = b[b.New_Stability.isin([stab])&b.k_degA.isin([kdegA])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    skA.iloc[i] = line
    line  = []
    i = i + 1

skA.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + ' - $k_{degA}$')
plt.yticks(range(0, 20, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $k_{degA}$ - Replating',  **titfont)
plt.savefig(path_fig + 'stability_kA_rep_' + circuit + '.png')


#DIAGRAMA DE BARRAS EN FUNCION DE STABILITY Y K_DEGB #
c = df4rep.groupby(by = ['New_Stability', 'k_degB']).size()
c = c.reset_index()

s = c['New_Stability'].unique().tolist()
kB = c['k_degB'].unique().tolist()
skB = pd.DataFrame(  index= s, columns= kB)

line = []
i = 0

for stab in s:
    for kdegB in kB:
        d = c[c.New_Stability.isin([stab])&c.k_degB.isin([kdegB])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    skB.iloc[i] = line
    line  = []
    i = i + 1

skB.plot(kind = 'bar', width=0.5, figsize=(4,5))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend(title = circuit_tit + ' - $k_{degB}$')
plt.yticks(range(0, 25, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '- $k_{degB}$ - Replating',  **titfont)
plt.savefig(path_fig + 'stability_kB_rep_' + circuit + '.png')





#####################             100 POSICIONES       ##################################

#Contar de los 100, cuantos son estables y cuantos inestables
index100, counts100 = np.unique(df100[['Stability']].to_numpy(), return_counts=True)
suma_stability100 = pd.Series(counts100,  index100)


# 100 POSICIONES REPLATING 09 #
#Contar de los 100, cuantos son estables y cuantos inestables
index100rep, counts100rep = np.unique(df100rep[['New_Stability']].to_numpy(), return_counts=True)
suma_stability100rep = pd.Series(counts100rep,  index100rep)


columnas = [suma_stability100, suma_stability100rep]

col = ['Without Perturbation ', 'With Perturbation']
indexes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
st100 = pd.DataFrame(index= indexes, columns= col)
line = []

for i in indexes:
    for exp in columnas:
        
        if exp.index.isin([i]).any():
            num = exp[i]
        else:
            num = 0
    
        line.append(num)
    
    
    ind = round(i * 10)
    
    st100.iloc[ind] = line
    line = []
    
plt.figure()   
st100.plot(kind = 'bar', width=0.5, figsize=(5,4))
plt.xlabel('Stability',  **axsfont)
plt.ylabel('Count',  **axsfont)
plt.legend()
plt.yticks(range(0, 45, 5))
plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)
#plt.title(circuit_tit + '\n $t_{deg}$ = ' + str(df100rep['t_deg_x'][0]) + ', $k_{degA}$ = ' + str(df100rep['k_degA'][0]) + ', $k_{degB}$ = ' + str(df100rep['k_degB'][0]), **titfont)

plt.savefig(path_fig + 'stability_100_wtvswthoutrep' + circuit + '.png')




#COMPARE STABILITY 1.0 vs STABILITY 0.0

# query1 = df100.query('Stability == 1.0')
# query0 = df100.query('Stability == 0.0')




# plt.figure()
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'olive' ]
# plt.grid(axis='y',linestyle= 'dotted' , color = 'k', linewidth=0.3)


# exp1 = query1.iloc[1]
# df1 = pd.DataFrame(columns=exp1.index.to_list())
# posS1 = exp1['posS1']
# posS1 = posS1.split('\n')
# posS2 = exp1['posS2']
# posS2 = posS2.split('\n')
# num_cells = [len(posS1), len(posS2)]  # len(posS3)

# #Strain 1
# for pos in posS1:
#     pos = re.findall(r'-?\d+\.?\d*', pos)
#     x = int(pos[0])
#     y = int(pos[1])
#     plt.plot(x,y, marker = 'o', color=colors[1], label = 'Strain 1')

# #Strain2
# for pos in posS2:
#     pos = re.findall(r'-?\d+\.?\d*', pos)
#     x = int(pos[0])
#     y = int(pos[1])
#     plt.plot(x,y, marker =  '^', color=colors[1], label = 'Strain 2')
    


# exp0 = query0.iloc[5]
# df0 = pd.DataFrame(columns=exp0.index.to_list())
# posS1 = exp0['posS1']
# posS1 = posS1.split('\n')
# posS2 = exp0['posS2']
# posS2 = posS2.split('\n')
# num_cells = [len(posS1), len(posS2)]  # len(posS3)

# #Strain 1
# for pos in posS1:
#     pos = re.findall(r'-?\d+\.?\d*', pos)
#     x = int(pos[0])
#     y = int(pos[1])
#     plt.plot(x,y, marker = 'o', color=colors[2], label = 'Strain 1')

# #Strain2
# for pos in posS2:
#     pos = re.findall(r'-?\d+\.?\d*', pos)
#     x = int(pos[0])
#     y = int(pos[1])
#     plt.plot(x,y, marker =  '^', color=colors[2], label = 'Strain 2')




perturb_anal = df100rep.drop(columns=['Unnamed: 0', 'Num_Cells_Strain1', 'Num_Cells_Strain2',
       'Num_Cells_Strain3', 't_deg_x', 'k_degA', 'k_degB', 'ratioS1',
       'ratioS2', 'ratioS3_S1', 'ratioS3_S2', 'posS1', 'posS2', 'posS3', 'Replating'])

#perturb_anal = df100rep.drop(columns=['Unnamed: 0', 'Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg',
  #     'k_degA', 'k_degB', 'ratioS1', 'ratioS2', 'posS1', 'posS2','Survivors'])

perturb_anal.to_csv('C:/Users/elenu/Documents/MASTER AI/TFM/Perturbation Analysis/' + circuit + '.csv' )
