import numpy as np
import tensorflow as tf
import keras
from sklearn import preprocessing
from keras.layers import Concatenate,Dense,BatchNormalization
from keras import Input, Model
import utils

def ensemble_builder(params,
                     hyper_params=None,
                     selected_classes=None,
                     full=True):
    input_layer = Input(shape=(params['dims']))
    class_dict=params['class_weights']
    if(selected_classes is None):
        selected_classes=list(range(params['n_cats']))   
    single_cls,loss,metrics=[],{},{}
    for i in selected_classes:
        nn_i=nn_builder(params=params,
                        hyper_params=hyper_params,
                        input_layer=input_layer,
                        i=i,
                        n_cats=params['n_cats'])
        single_cls.append(nn_i)
        loss[f'out_{i}']=weighted_loss(specific=i,
                                       class_dict=class_dict)
        metrics[f'out_{i}']= 'accuracy'
    if(full):
        k=len(selected_classes)
        nn_k=nn_builder(params=params,
                        hyper_params=hyper_params,
                        input_layer=input_layer,
                        i=k,
                        n_cats=params['n_cats'])
        single_cls.append(nn_k)
        loss[f'out_{k}']=weighted_loss(specific=None,
                                       class_dict=class_dict)
        metrics[f'out_{k}']= 'accuracy'
    model= Model(inputs=input_layer, 
                 outputs=single_cls)
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=metrics,
                  jit_compile=False)
    return model

def single_builder(params,
                   hyper_params=None):
    input_layer = Input(shape=(params['dims']))
    class_dict=params['class_weights']
    nn=nn_builder(params=params,
                    hyper_params=hyper_params,
                    input_layer=input_layer,
                    i=0,
                    n_cats=params['n_cats'])
    loss=weighted_loss(specific=None,
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

def weighted_loss(specific,class_dict):
    n_cats=len(class_dict)
    class_weights=np.zeros(n_cats,dtype=np.float32)
    for i in range(n_cats):
        class_weights[i]=class_dict[i] #1.0/class_dict[i]
    if(not (specific is None)):
        class_weights[specific]*=  (len(class_dict)/2)
    return keras_loss(class_weights)

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

def get_callback(callback_type):
    if(callback_type=="min"):
        return MinAccEarlyStopping
    if(callback_type=="all"):
        return ImprovEarlyStopping
    if(callback_type=="total"):
        return TotalEarlyStopping
    raise Exception(f"Unknow callback type{callback_type}")

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

class MinAccEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=15,
                       verbose=1):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.verbose=verbose

    def init(self,n_clfs):
        pass 

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        acc=[]
        for key_i in logs.keys():
            if("accuracy" in key_i):
                acc.append(logs[key_i])
        min_acc=np.amin(acc)
        if(min_acc>self.best):
            if(self.verbose):
                print(f"min_acc{min_acc},{self.wait}")
            self.best=min_acc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait+=1
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)


class ImprovEarlyStopping(keras.callbacks.Callback):
    def __init__(self,#n_clfs, 
                      patience=15,
                      eps=0.0001,
                      good_acc=0.95,
                      verbose=0):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.eps=eps
        self.good_acc=good_acc
        self.verbose=verbose
        
    def init(self,n_clfs):
        self.best=np.zeros(n_clfs,dtype=float)
        self.wait=np.zeros(n_clfs,dtype=int)
    
    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        for key_i in logs.keys():
            if("accuracy" in key_i):
                i=utils.extract_number(key_i)
                if(self.best[i]>self.good_acc):
                    self.wait[i]=self.patience+1
                    continue
                current_i=logs[key_i]
                diff_i= current_i-self.best[i]
                if(diff_i>= self.eps):
                    self.best[i]=current_i
                    self.wait[i]=0
                else:
                    self.wait[i]+=1
        if(self.verbose):
            print(f"epoch:{epoch}")
            print(self.best)
            print(self.wait)
        min_wait=np.amin(self.wait)
        if(min_wait>self.patience):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
        else:
            self.best_weights = self.model.get_weights()


class TotalEarlyStopping(keras.callbacks.Callback):
    def __init__(self,#n_clfs, 
                      patience=15,
                      eps=0.0001,
                      good_acc=0.95,
                      verbose=0):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.eps=eps
        self.good_acc=good_acc
        self.verbose=verbose
    
    def init(self,n_clfs):
        self.best=0
        self.wait=0

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        total_acc=0.0
        for key_i in logs.keys():
            if("accuracy" in key_i):
                total_acc+=logs[key_i]
        diff= total_acc-self.best
        if(diff>= self.eps):
            self.best=total_acc
            self.wait=0
        else:
            self.wait+=1
        if(self.verbose):
            print(f"epoch:{epoch}")
            print(self.best)
            print(self.wait)
        if(self.wait>self.patience):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
        else:
            self.best_weights = self.model.get_weights()