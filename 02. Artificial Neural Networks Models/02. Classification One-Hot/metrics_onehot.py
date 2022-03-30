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

attributes = pd.read_csv(path + 'training_dataset_clean.csv')
#labels = pd.read_csv(path + 'training_dataset_onehot_2clases.csv')
labels = pd.read_csv(path + 'training_dataset_continuous_2labels.csv')
#labels = pd.read_csv(path + 'training_dataset_continuous.csv')

#attributes = pd.read_csv(path + 'data_replating_clean.csv')
#labels = pd.read_csv(path + 'data_replating_onehot_2clases.csv')
#labels = pd.read_csv(path + 'data_replating_continuous_2labels.csv')

#model = load_model(path_models + 'model_binary_replating.h5', custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})
model = load_model(path_models + 'model_binary.h5', custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})

#model = load_model(path_models + 'model_classification_onehot_replating.h5')
#model = load_model(path_models + 'model_classification_onehot.h5')

#model = load_model(path_models + 'model_regression.h5', custom_objects={'precision_m': precision_m})


# hacemos un primer split para generar los valores de train y preparar el 20% restante para su division a 10%-10%
# controlamos que los datos de train y test sean siempre los mismos para controlar resultados mediante random_state
train_prop = 0.8
dev_prop = 0.1
test_prop = 0.1
    
x_train, x_rest, t_train, t_rest = train_test_split(attributes, labels, train_size = train_prop, random_state = 1, stratify = labels )

# hacemos el segundo spplit para generar dev y test

x_dev, x_test, t_dev, t_test = train_test_split(x_rest, t_rest, test_size = 0.5, random_state = 1, stratify = t_rest )


print('Training:', model.evaluate(x_train, t_train))
print('Validation:', model.evaluate(x_dev, t_dev))
print('Test:', model.evaluate(x_test, t_test))








### PARA CALCULAR F1, PRECISION Y RECALL EN ONE-HOT
pred_train = np.round(model.predict(x_train))
index = x_train.index.tolist()
t_train['index'] = t_train.index

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(pred_train)):
    
    pred = pred_train[i].tolist()
    
    query= 'index ==' + str(index[i])
    target = [float(t_train.query(query).Unstable), float(t_train.query(query).Stable)]
    
    if pred ==  [0.0, 1.0] and target ==  [0.0, 1.0]:
        tp = tp + 1
    elif pred ==  [0.0, 1.0] and target ==  [1.0, 0.0]:
        fp = fp + 1
    elif pred ==  [1.0, 0.0] and target ==  [1.0, 0.0]:
        tn = tn + 1
    elif pred ==  [1.0, 0.0] and target ==  [0.0, 1.0]:
        fn = fn + 1


prec = tp / (tp + fp)
rec = tp / (tp + fn)
accuracy = (tp + fn) / (tp + tn + fp + fn)
f1 = (2 * prec * rec)/(prec + rec)

print('training dataset \n:')
print('precision:', prec)
print('recall:', rec)
print('accuracy:', accuracy)
print('f1:', f1)


pred_val = np.round(model.predict(x_dev))
index = x_dev.index.tolist()
t_dev['index'] = t_dev.index

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(pred_val)):
    
    pred = pred_val[i].tolist()
    
    query= 'index ==' + str(index[i])
    target = [float(t_dev.query(query).Unstable), float(t_dev.query(query).Stable)]
    
    if pred ==  [0.0, 1.0] and target ==  [0.0, 1.0]:
        tp = tp + 1
    elif pred ==  [0.0, 1.0] and target ==  [1.0, 0.0]:
        fp = fp + 1
    elif pred ==  [1.0, 0.0] and target ==  [1.0, 0.0]:
        tn = tn + 1
    elif pred ==  [1.0, 0.0] and target ==  [0.0, 1.0]:
        fn = fn + 1


prec = tp / (tp + fp)
rec = tp / (tp + fn)
accuracy = (tp + fn) / (tp + tn + fp + fn)
f1 = (2 * prec * rec)/(prec + rec)

print('validation dataset \n:')
print('precision:', prec)
print('recall:', rec)
print('accuracy:', accuracy)
print('f1:', f1)


pred_test = np.round(model.predict(x_test))
index = x_test.index.tolist()
t_test['index'] = t_test.index

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(pred_test)):
    
    pred = pred_test[i].tolist()
    
    query= 'index ==' + str(index[i])
    target = [float(t_test.query(query).Unstable), float(t_test.query(query).Stable)]
    
    if pred ==  [0.0, 1.0] and target ==  [0.0, 1.0]:
        tp = tp + 1
    elif pred ==  [0.0, 1.0] and target ==  [1.0, 0.0]:
        fp = fp + 1
    elif pred ==  [1.0, 0.0] and target ==  [1.0, 0.0]:
        tn = tn + 1
    elif pred ==  [1.0, 0.0] and target ==  [0.0, 1.0]:
        fn = fn + 1


prec = tp / (tp + fp)
rec = tp / (tp + fn)
accuracy = (tp + fn) / (tp + tn + fp + fn)
f1 = (2 * prec * rec)/(prec + rec)

print('test dataset \n:')
print('precision:', prec)
print('recall:', rec)
print('accuracy:', accuracy)
print('f1:', f1)