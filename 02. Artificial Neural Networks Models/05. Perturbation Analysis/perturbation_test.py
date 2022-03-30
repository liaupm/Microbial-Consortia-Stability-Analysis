# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:30:34 2021

@author: elenu
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines


mpl.rc('font',family='Times New Roman')
titfont = {'fontname': 'Times New Roman', 'weight': 'bold', 'fontsize':'15'}
axsfont = {'fontname': 'Times New Roman', 'fontsize':'12'}



path = 'C:/Users/elenu/Documents/MASTER AI/TFM/Data ANN/Testing/'
df = pd.read_csv(path +'data_replating.csv')

#DIAGRAMA DE BARRAS EN FUNCION DE NEW_STABILITY Y SURVIVORS #
a = df.groupby(by = ['New_Stability', 'Survivors']).size()
a = a.reset_index()

s = a['New_Stability'].unique().tolist()
t = a['Survivors'].unique().tolist()
st = pd.DataFrame(  index= s, columns= t)

line = []
i = 0

for stab in s:
    for surv in t:
        d = a[a.New_Stability.isin([stab])&a.Survivors.isin([surv])][0].tolist()
        if d:
            line.append(d[0])
        else:
            line.append(0)
    
    st.iloc[i] = line
    line  = []
    i = i + 1


st.to_csv(path + 'perturbation_analysis.csv')