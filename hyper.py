import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Input, Model
import keras_tuner as kt
import argparse
import data

class MultiKTBuilder(object): 
    def __init__(self,params):#,hidden=[(0.25,5),(0.25,5)]):
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

#def single_exp(data_path,hyper_path=None,n_iter=5):
#    df=data.from_arff(data_path)
#    X,y=data.prepare_data(df,target=-1)
#    data_params=data.get_dataset_params(X,y)
#    print(data_params)
#    best=bayes_optim(X,y,data_params,n_iter)
#    best=[tools.round_data(best_i,4) for best_i in best]
#    with open(hyper_path,"a") as f:
#        f.write(f'{str(best)}\n') 
#    return best

def bayes_optim(exp,n_iter=5,verbose=1):
    model_builder= MultiKTBuilder(exp.params) 

    tuner=kt.BayesianOptimization(model_builder,
                objective='val_loss',
                max_trials=n_iter,
                overwrite=True)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5)
    x_train,y_train=exp.get_train()
    x_valid,y_valid=exp.get_valid()
    tuner.search(x=x_train, 
                 y=y_train, 
                 epochs=150,
                 batch_size=exp.params['batch'], 
                 validation_data=(x_valid, y_valid),
                 verbose=verbose,
                 callbacks=[stop_early])
    
    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
    best=  best_hps.values
    return best
#    relative={key_i: (value_i/data_params['dims'])  
#                for key_i,value_i in best.items()
#                    if('unit' in key_i)}
#    models=tuner.get_best_models()
#    acc=get_metric_value(tuner,X,y)
#    return best,relative,acc

if __name__ == '__main__':
    single_exp('raw/mfeat-factors.arff')