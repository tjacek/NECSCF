import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from keras.layers import Concatenate,Dense,BatchNormalization
from keras import Input, Model
import utils

def ensemble_builder(params,
                     hyper_params=None,
                     class_dict=None,
                     loss_gen=None,
                     full=True):
    if(loss_gen is None):
        loss_gen=WeightedLoss()
    input_layer = Input(shape=(params['dims']))
    if(class_dict is None):
        class_dict=params['class_weights']
    n_cats=params['n_cats']
    single_cls,loss,metrics=[],{},{}
    for i in range(n_cats):
        nn_i=nn_builder(params=params,
                        hyper_params=hyper_params,
                        input_layer=input_layer,
                        i=i,
                        n_cats=params['n_cats'])
        single_cls.append(nn_i)
        loss[f'out_{i}']=loss_gen(specific=i,
                                  class_dict=class_dict)
        metrics[f'out_{i}']= 'accuracy'
    if(full):
        nn_k=nn_builder(params=params,
                        hyper_params=hyper_params,
                        input_layer=input_layer,
                        i=n_cats,
                        n_cats=n_cats)
        single_cls.append(nn_k)
        loss[f'out_{n_cats}']=loss_gen(specific=None,
                                       class_dict=class_dict)
        metrics[f'out_{n_cats}']= 'accuracy'
    model= Model(inputs=input_layer, 
                 outputs=single_cls)
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=metrics,
                  jit_compile=False)
    return model

def single_builder(params,
                   hyper_params=None,
                   class_dict=None):
    input_layer = Input(shape=(params['dims']))
    if(class_dict is None):
        class_dict=params['class_weights']
    nn=nn_builder(params=params,
                    hyper_params=hyper_params,
                    input_layer=input_layer,
                    i=0,
                    n_cats=params['n_cats'])
    
    loss=WeightedLoss()(specific=None,
                       class_dict=class_dict)
    model= Model(inputs=input_layer, 
                 outputs=nn)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def nn_builder(params,
               hyper_params,
               input_layer=None,
               i=0,
               n_cats=None):
    if(input_layer is None):
        input_layer = Input(shape=(params['dims']))
    if(n_cats is None):
        n_cats=params['n_cats']
    x_i=input_layer
    for j in range(hyper_params['layers']):
        hidden_j=int(params['dims'][0]* hyper_params[f'units_{j}'])
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
    if(hyper_params['batch']):
        x_i=BatchNormalization(name=f'batch_{i}')(x_i)
    x_i=Dense(n_cats, activation='softmax',name=f'out_{i}')(x_i)
    return x_i

class WeightedLoss(object):
    def __init__(self,multi=True):
        self.multi=multi

    def init(self,data):
        pass
    
    def __call__(self,specific,class_dict):
        if(self.multi):
            n_cats=len(class_dict)
            class_weights=np.zeros(n_cats,dtype=np.float32)
            for i in range(n_cats):
                class_weights[i]=class_dict[i]
            if(not (specific is None)):
                class_weights[specific]*=  (len(class_dict)/2)
            return keras_loss(class_weights)
        else:
            class_dict=class_dict.copy()
            if(not (specific is None)):
                class_dict[specific]*=(len(class_dict)/2)
            return class_dict

    def __str__(self):
        name="class_ens"
        if(not self.multi):
            name=f"separ_{name}"
        return name

@keras.saving.register_keras_serializable(name="weighted_loss")
def keras_loss( class_weights):
    def loss(y_obs,y_pred):        
        y_obs = tf.dtypes.cast(y_obs,tf.int32)
        hothot=  tf.dtypes.cast( y_obs,tf.float32)
        weights = tf.math.multiply(class_weights,hothot)
        weights = tf.reduce_sum(weights,axis=-1)
        y_obs= tf.argmax(y_obs,axis=1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=y_obs, 
            logits=y_pred,
            weights=weights
        )
        return losses
    return loss