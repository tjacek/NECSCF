import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Dense,BatchNormalization,Concatenate
from tensorflow.keras import Input, Model
#import loss

#class NeuralEnsemble(object):
#    def __init__(self,model,params,hyper_params):
#        self.model=model
#        self.params=params
#        self.hyper_params=hyper_params
#        self.pred_models=None
#        self.extractors=None



def ensemble_builder(params,hyper_params):
    input_layer = Input(shape=(params['dims']))
    class_dict=params['class_weights']
    single_cls,loss,metrics=[],{},{}
    for i in range(params['n_cats']):
        nn_i=nn_builder(params=params,
                        hyper_params=hyper_params,
                        input_layer=input_layer,
                        as_model=False,
                        i=i,
                        n_cats=params['n_cats'])
        single_cls.append(nn_i)
        loss[f'out_{i}']=weighted_loss(i=i,
                                       class_dict=class_dict,
                                       alpha=0.5)
        metrics[f'out_{i}']= 'accuracy'
    model= Model(inputs=input_layer, outputs=single_cls)
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=metrics)
    return model

def nn_builder(params,hyper_params,input_layer=None,as_model=True,i=0,n_cats=None):
    print(hyper_params)
    if(input_layer is None):
        input_layer = Input(shape=(params['dims']))
    if(n_cats is None):
        n_cats=params['n_cats']
    x_i=input_layer#Concatenate()([common,input_i])
#    for j,hidden_j in enumerate(hyper_params['layers']):
    for j in range(hyper_params['layers']):
        hidden_j=hyper_params[f'units_{j}']
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
    if(hyper_params['layers']):
        x_i=BatchNormalization(name=f'batch_{i}')(x_i)
    x_i=Dense(n_cats, activation='softmax',name=f'out_{i}')(x_i)
    if(as_model):
        return Model(inputs=input_layer, outputs=x_i)
    return x_i

def weighted_loss(i,class_dict,alpha=0.5):
    one_i=class_dict[i]
    other_i=sum(class_dict.values())-one_i
    cat_size_i  = alpha*(1/one_i)
    other_size_i= (1.0-alpha) * (1/other_i)
    class_weights= [other_size_i for i in range(len(class_dict) )]
    class_weights[i]=cat_size_i
    class_weights=np.array(class_weights,dtype=np.float32)
    return keras_loss(class_weights)

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