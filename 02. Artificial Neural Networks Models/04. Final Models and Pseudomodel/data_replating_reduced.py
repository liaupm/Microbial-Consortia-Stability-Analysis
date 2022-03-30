import pandas as pd

path = 'C:/Users/elenu/Documents/MASTER AI/TFM/Data ANN/Testing/'

data_rep = pd.read_csv(path + 'data_replating.csv')



data_rep_stable = data_rep.query('New_Stability == 1.0')


data_rep_reduced = data_rep_stable.drop_duplicates(subset=['Num_Cells_Strain1', 'Num_Cells_Strain2', 't_deg', 'k_degA', 'k_degB',
       'ratioS1', 'ratioS2', 'Stability'], keep="last")

data_rep_reduced.to_csv(path + 'data_replating_reduced.csv')