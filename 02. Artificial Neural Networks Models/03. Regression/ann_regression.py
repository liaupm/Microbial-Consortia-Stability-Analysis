import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import random
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import multiprocessing as mp





#METRICAS
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




def ann(cleanattributes, onehot, df, n_epochs, learning_rate, batch_size, n_neurons_per_hlayer, dropout, act_fun, my_regularizer, my_initializer, my_optimizer, norm, n):
    
    print('ANN ' + str(n) + ' empezada')
    
    attributes = pd.read_csv(cleanattributes)
    labels = pd.read_csv(onehot)
    
    train_prop = 0.8
    dev_prop = 0.1
    test_prop = 0.1
    
    # hacemos un primer split para generar los valores de train y preparar el 20% restante para su division a 10%-10%
    # controlamos que los datos de train y test sean siempre los mismos para controlar resultados mediante random_state

    x_train, x_rest, t_train, t_rest = train_test_split(attributes, labels, train_size = train_prop, random_state = 1, stratify = labels )

    # hacemos el segundo spplit para generar dev y test

    x_dev, x_test, t_dev, t_test = train_test_split(x_rest, t_rest, test_size = 0.5, random_state = 1, stratify = t_rest )

    #parametros usados para la capa de entrada y salida
    input_size_neuralnetwork = x_train.shape[1]
    output_size_neuralnetwork = t_train.shape[1]

    #hiperparametros de la red neuronal
    #n_epochs, learning_rate, batch_size, n_neurons_per_hlayer, dropout, my_initializer, act_fun, my_regularizer, norm
    
    
    # MODELO
    model = keras.Sequential(name="DeepFeedforward")
    model.add(keras.layers.InputLayer(input_shape=(input_size_neuralnetwork,), batch_size=None))
    
    dropout_rate = [dropout, dropout, dropout, dropout]
    
    
    if my_optimizer == 'sgd_momentum':
        opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
    elif my_optimizer == 'sgd_nesterov':
        opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9,nesterov=True)
    elif my_optimizer == 'rms_prop':
        opt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif my_optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    else:
        opt = tf.keras.optimizers.SGD(lr=learning_rate)

    
    if my_regularizer == 'l1':
        reg = keras.regularizers.l1(0.001)
    elif my_regularizer == 'l2':
        reg = keras.regularizers.l2(0.001)
    else:
        reg = None
        
    
    if my_initializer == 'he_uniform':
        init = keras.initializers.he_uniform(seed=None)
    elif my_initializer == 'zeros':
        init = keras.initializers.Zeros()
    elif my_initializer == 'random_uniform':
        init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    elif my_initializer == 'random_normal':
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    elif my_initializer == 'he_normal':
        init = keras.initializers.he_normal(seed=None)
        
        
    for neurons, d in zip(n_neurons_per_hlayer, dropout_rate):
        model.add(keras.layers.Dense(neurons, activation = act_fun, kernel_initializer = init, kernel_regularizer = reg))
        model.add(keras.layers.Dropout(rate=d))
        
        if norm == 1:
            model.add(keras.layers.BatchNormalization()) #Normalization after
    
    model.add(keras.layers.Dense(output_size_neuralnetwork, activation="sigmoid"))
    
    
    # COMPILE
    #usaremos crossentropy y adagrad por ahora
    model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer= opt,
              metrics=["accuracy", f1_m, precision_m, recall_m])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='auto')
    
    for repeat in range(5):
        
        # TRAIN
    
        history = model.fit(x_train, t_train, batch_size = batch_size, epochs = n_epochs, verbose=0, validation_data=(x_dev, t_dev), callbacks = early_stop)
       
        # RESULTS
        results=pd.DataFrame(history.history)
        
        train_loss = history.history['loss']
        validation_loss = history.history['val_loss']
        
        train_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']
        
        
        acc_train = results.accuracy.values[-1:][0]
        acc_val =  results.val_accuracy.values[-1:][0]
        #bias = round((1 - results.categorical_accuracy.values[-1:][0])*100, 1)
        #var = round((1 - results.val_categorical_accuracy.values[-1:][0])*100, 1)
       
        # TEST
        resultados_test = model.evaluate(x_test, t_test)
        loss_test = resultados_test[0]
        acc_test = resultados_test[1]
        f1 = resultados_test[2]
        prec = resultados_test[3]
        rec = resultados_test[4]
        
   

        df = df.append({'Epochs': n_epochs, 'Learning_Rate': learning_rate, 'Batch_Size': batch_size, 'Neurons': n_neurons_per_hlayer, 'Dropout': dropout, 'Activation': act_fun, 'Initializer': my_initializer, 'Regularizer':  my_regularizer, 'Optimizer': my_optimizer, 'Normalization': norm, 'Acc_train': acc_train, 'Acc_val': acc_val, 'Acc_test': acc_test, 'f1': f1, 'Precision': prec, 'Recall': rec}, ignore_index=True)
        
       
    with open( 'C:/TFM/ann/Regression/data_ann_' + str(n) + '.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    print('ANN ' + str(n) + ' acabada')
    
    












if __name__ == '__main__':
    

    
    start_time = time.time()
    pool = mp.Pool()
    
    
    cleanattributes = 'C:/TFM/ann/training_dataset_clean.csv'
    onehot = 'C:/TFM/ann/training_dataset_continuous.csv'
    
    df = pd.DataFrame(columns=['Epochs', 'Learning_Rate', 'Batch_Size', 'Neurons', 'Dropout', 'Activation', 'Initializer', 'Regularizer', 'Normalization', 'Acc_train', 'Acc_val', 'Acc_test', 'f1', 'Precision', 'Recall'])
    
    n_epochs = 3000
    lr= [0.0001, 0.0005, 0.001]
    batch = [512, 1024] #, 5018]
    n_neurons_per_hlayer = [125, 100, 75, 50, 25]
    dr = [0.25, 0.1, 0.0]
    act = ['relu']
    regularizer = ['l1', 'l2', 'None']
    initializer = ['he_uniform', 'he_normal', 'random_uniform', 'random_normal'] # 'zeros'
    optimizer = ['adam', 'rms_prop', 'sgd', 'sgd_momentum', 'sgd_nesterov']
    
    
    normalization = [1, 0]
    
    n = 1
    
    for learning_rate in lr:
        
        for batch_size in batch:
            
            for dropout in dr:
                
                for my_initializer in initializer:
                    
                    for my_regularizer in regularizer:
                        
                        for act_fun in act:
                            
                            for norm in normalization:
                                
                                for my_optimizer in optimizer:
                        
                                    #ann(cleanattributes, onehot, df, n_epochs, learning_rate, batch_size, n_neurons_per_hlayer, dropout, act_fun, my_regularizer, my_initializer, norm, n)
                                    pool.apply_async(ann, args = ( cleanattributes, onehot, df, n_epochs, learning_rate, batch_size, n_neurons_per_hlayer, dropout, act_fun, my_regularizer, my_initializer, my_optimizer, norm, n )) 
                                    n = n + 1

        
    pool.close()
    pool.join()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
    







