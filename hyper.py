import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Input, Model
import keras_tuner as kt
import argparse
import data

class MultiKTBuilder(object): 
    def __init__(self,params):
        self.params=params
        self.hidden=(1,10)

    def __call__(self,hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int('layers', 1, 2)):
            hidden_i=hp.Int('units_' + str(i), 
                    min_value= int(self.params['dims']*self.hidden[0]), 
                    max_value=int(self.params['dims']*self.hidden[1]),
                    step=10)
            model.add(tf.keras.layers.Dense(units=hidden_i ))
        batch=hp.Choice('batch', [True, False])
        if(batch):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(self.params['n_cats'], activation='softmax'))
        model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_hyper(self,tuner):
        best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
        best=  best_hps.values
        return best

class EffBuilder(object):
    def __init__(self,params):
        self.params=params
        self.first=(1,10)
        self.second=(3,20)

    def __call__(self,hp):
        model = tf.keras.Sequential()
        dims=int(self.params['dims'])
        first_hp=hp.Int('units_0', 
                        min_value= dims*self.first[0], 
                        max_value=dims*self.first[1],
                        step=10)
        model.add(tf.keras.layers.Dense(units=first_hp))
        n_cats=int(self.params['n_cats'])
        second_hp=hp.Int('units_1', 
                          min_value=n_cats*self.second[0], 
                          max_value=n_cats*self.second[1],
                          step=10)
        model.add(tf.keras.layers.Dense(units=second_hp))
        batch=hp.Choice('batch', [True, False])
        if(batch):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(self.params['n_cats'], 
                                        activation='softmax'))
        model.compile('adam', 
                      'sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def extract_hyper(self,tuner):
        best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
        best=  best_hps.values
        best['layers']=2
        return best

def bayes_optim(exp,n_iter=5,verbose=1):
#    model_builder= MultiKTBuilder(exp.params) 
    model_builder= EffBuilder(exp.params) 

    tuner=kt.BayesianOptimization(model_builder,
                objective='val_loss',
                max_trials=n_iter,
                overwrite=True)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5)
    x_train,y_train=exp.split.get_train()
    x_valid,y_valid=exp.split.get_valid()
    tuner.search(x=x_train, 
                 y=y_train, 
                 epochs=150,
                 batch_size=exp.params['batch'], 
                 validation_data=(x_valid, y_valid),
                 verbose=verbose,
                 callbacks=[stop_early])
    
    tuner.results_summary()
    return model_builder.extract_hyper(tuner)

if __name__ == '__main__':
    single_exp('raw/mfeat-factors.arff')