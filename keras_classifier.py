# -*- coding: utf-8 -*-
"""
NICE EXAMPLE OF A CLASSIFIER WITH OPTIONS FOR VARIOUS REGULARISATION METHODS

Created on Tue Mar 13 10:02:48 2018

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


now = datetime.now()
K.clear_session()


#load in datasets - these are output from the classification stage
#REMEMBER - THESE ARE SCALED - CAN USE COL_STATS TO UNSCALE THEM

train_dataset = pd.read_csv("train.csv")
valid_dataset = pd.read_csv("valid.csv")

X_train = np.float32(train_dataset).values
X_valid = np.float32(valid_dataset).values

Y_train = train_dataset["IAP_purchaser"]
Y_valid = valid_dataset["IAP_purchaser"]

#one-hot encode the dependent variable
Y_train = keras.utils.np_utils.to_categorical(Y_train)
Y_valid = keras.utils.np_utils.to_categorical(Y_valid)


#%%

#network parameters (will be tuned)
layer_1_nodes=50
layer_2_nodes=50
layer_3_nodes=50
layer_4_nodes=15
layer_5_nodes=5


l1=1e-5
l2_lambda=1e-5
learning_rate=0.0001
beta1=0.99
beta2=0.999
dropout=0.5
decay=0.0

activation_fn = "sigmoid"
act=keras.layers.ELU(alpha=1.0)

nb_epoch = 1000
batch_size = 100

initialiser = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

#%%
#setting up the network
input_dim = X_train.shape[1]

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
#neural_net.add( Dense(layer_4_nodes,activation=activation_fn))
#neural_net.add(Dropout(dropout))
#neural_net.add( Dense(layer_5_nodes,activation=activation_fn))
#neural_net.add(Dropout(dropout))
neural_net.add(Dense(2,activation='softmax',init=initialiser))



adam=optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=None, decay=decay, amsgrad=False)
sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True)

#%%
#run the network


#NEED TO CONSIDER THESE METRICS
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