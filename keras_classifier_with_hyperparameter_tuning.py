# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:07:40 2018

@author: alex.hall
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:41:55 2018

@author: alex.hall
"""



#backpropogation neural net with dropout


from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import optimizers
from keras.models import Model, load_model,Sequential
from keras import backend as K
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from hyperopt import space_eval


now = datetime.now()
K.clear_session()


#load in datasets - these are output from the classification stage
#REMEMBER - THESE ARE SCALED - CAN USE COL_STATS TO UNSCALE THEM

train_dataset = pd.read_csv("\\\\nas01\\analytics\\Analytics Projects and Results\\LTV prediction\\csv_datasets\\classification_input\\reduced\\train.csv")
valid_dataset = pd.read_csv("\\\\nas01\\analytics\\Analytics Projects and Results\\LTV prediction\\csv_datasets\\classification_input\\reduced\\valid.csv")
col_stats= pd.read_csv("\\\\nas01\\analytics\\Analytics Projects and Results\\LTV prediction\\csv_datasets\\classification_input\\reduced\\col_stats.csv")

X_train = np.float32(train_dataset.drop(["amplitude_id","IAP_purchaser"],axis=1).values)
X_valid = np.float32(valid_dataset.drop(["amplitude_id","IAP_purchaser"],axis=1).values)

Y_train = train_dataset["IAP_purchaser"]
Y_valid = valid_dataset["IAP_purchaser"]

#one-hot encode the dependent variable
Y_train = keras.utils.np_utils.to_categorical(Y_train)
Y_valid = keras.utils.np_utils.to_categorical(Y_valid)


#%%

#network parameters (for tuning)
layer_1_nodes=50
layer_2_nodes=50
layer_3_nodes=50
layer_4_nodes=15
layer_5_nodes=5


l1=1e-5
l2_lambda=1e-5
#learning_rate=0.0001
#beta1=0.99
#beta2=0.999
#dropout=0.5
decay=0.0

activation_fn = "sigmoid"
act=keras.layers.ELU(alpha=1.0)

nb_epoch = 100
#batch_size = 100

initialiser = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)


space = {'dropout' : hp.uniform('dropout', .25 , .75),
         'batch_size' : hp.choice('batch_size', np.arange(24, 256, dtype=int)),
         'layer_1_nodes' : hp.choice('layer_1_nodes', np.arange(10, 50, dtype=int)),
         'layer_2_nodes' : hp.choice('layer_2_nodes', np.arange(10, 50, dtype=int)),
         'layer_3_nodes' : hp.choice('layer_3_nodes', np.arange(10, 50, dtype=int)),
         'layer_4_nodes' : hp.choice('layer_4_nodes', np.arange(10, 50, dtype=int)),
         'layer_5_nodes' : hp.choice('layer_5_nodes', np.arange(10, 50, dtype=int)),
         'learning_rate' : hp.loguniform('learning_rate',-15,-2),
         'beta1' : hp.uniform('beta1',0.95,0.9999),
         'beta2' : hp.uniform('beta2',0.99,0.9999),
         'activation_fn' : hp.choice('activation_fn',['relu','sigmoid','tanh'])}

#%%
#setting up the network
input_dim = X_train.shape[1]


#%%
# function to run the network

def make_neural_net(params):
    
     #add this to a layer to introduce l2 regularisation ,kernel_regularizer=regularizers.l2(l2_lambda)
    
    adam=optimizers.Adam(lr=params['learning_rate'], beta_1=params['beta1'], beta_2=params['beta2'], epsilon=None, decay=0, amsgrad=False)
   

    neural_net=Sequential()
    neural_net.add(Dense(params['layer_1_nodes'] , input_dim=input_dim , activation=params['activation_fn'],init=initialiser ))
    #neural_net.add(act)
    neural_net.add(Dropout(params['dropout']))
    neural_net.add( Dense(params['layer_2_nodes'] , activation=params['activation_fn'] ,init=initialiser))
    #neural_net.add(act)
    neural_net.add(Dropout(params['dropout']))
    neural_net.add( Dense(params['layer_3_nodes'] , activation=params['activation_fn'] ,init=initialiser))
    neural_net.add(Dropout(params['dropout']))
    neural_net.add( Dense(params['layer_4_nodes'] , activation=params['activation_fn'] , init=initialiser))
    neural_net.add(Dropout(params['dropout']))
    neural_net.add( Dense(params['layer_5_nodes'] , activation=params['activation_fn'] , init=initialiser))
    neural_net.add(Dropout(params['dropout']))
    neural_net.add(Dense(2 , activation='softmax' , init=initialiser))


    #NEED TO CONSIDER THESE METRICS
    neural_net.compile(optimizer=adam, 
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=0, mode='auto')

    neural_net.fit(x=X_train, y=Y_train,
                    epochs=nb_epoch,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    validation_data=(X_valid, Y_valid),
                    verbose=0,
                    callbacks=[early_stop])
                         
    pred_auc = neural_net.predict(X_valid,batch_size=100)
    auroc = roc_auc_score(Y_valid, pred_auc)
    
    print("AUROC:" , auroc)
    return{'loss': -auroc , 'status' : STATUS_OK}
                         
     
#%%
  #train hyperparameters 
trials=Trials()
best=fmin(make_neural_net , space , algo = tpe.suggest , max_evals = 20 , trials = trials )
print ('best = ')
print(space_eval(space,best)) #stupid feature of the hyperopt library - have to use space eval to see the values


#%%
#run and evaluate best model (include tensorboard here)


activation_fn = space_eval(space,best)['activation_fn']
batch_size = space_eval(space,best)['batch_size']
beta1 = best['beta1']
beta2=best['beta2']
dropout=best['dropout']
layer_1_nodes=space_eval(space,best)['layer_1_nodes']
layer_2_nodes=space_eval(space,best)['layer_2_nodes']
layer_3_nodes=space_eval(space,best)['layer_3_nodes']
layer_4_nodes=space_eval(space,best)['layer_4_nodes']
layer_5_nodes=space_eval(space,best)['layer_5_nodes']
learning_rate = best['learning_rate']

#add this to a layer to introduce l2 regularisation ,kernel_regularizer=regularizers.l2(l2_lambda)

neural_net=Sequential()
neural_net.add(Dense(layer_1_nodes,input_dim=input_dim,activation=activation_fn,init=initialiser ))
#neural_net.add(act)
neural_net.add(Dropout(dropout))
neural_net.add( Dense(layer_2_nodes,activation=activation_fn,init=initialiser))
#neural_net.add(act)
neural_net.add(Dropout(dropout))
neural_net.add( Dense(layer_3_nodes,activation=activation_fn,init=initialiser))
neural_net.add(Dropout(dropout))
neural_net.add( Dense(layer_4_nodes,activation=activation_fn,init=initialiser))
neural_net.add(Dropout(dropout))
neural_net.add( Dense(layer_5_nodes,activation=activation_fn,init=initialiser))
neural_net.add(Dropout(dropout))
neural_net.add(Dense(2,activation='softmax',init=initialiser))

adam=optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=None, decay=decay, amsgrad=False)


neural_net.compile(optimizer=adam, 
                    loss="binary_crossentropy",
                    metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                              save_best_only=False)
tensorboard = TensorBoard(log_dir='.\logs'+ now.strftime("%Y%m%d-%H%M%S") + "/",
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = neural_net.fit(x=X_train, y=Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_data=(X_valid, Y_valid),
                    verbose=1,
                    callbacks=[tensorboard,checkpointer]).history 
                         
classifier=load_model('model.h5')
predictions=classifier.predict(X_valid)

Y_true =[ np.where(r>0.5)[0][0] for r in Y_valid ]
predictions_decoded = [ np.where(r>0.5)[0][0] for r in predictions ]

confusion_matrix(Y_true,predictions_decoded)
