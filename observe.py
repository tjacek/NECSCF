import numpy as np
import tensorflow as tf
from keras import Input, Model
from collections import defaultdict
import matplotlib.pyplot as plt
import base,dataset,deep,ens

def show_loss(data_path,
              split_path,
              n_epochs=100):
    data_split=base.read_data_split(data_path=data_path,
		                            split_path=split_path)
    loss_dict={'base':base_loss,'custom':custom_loss}
    history_dict={key_i:[] for key_i in loss_dict}
    for i,data_i in data_split.selection_iter(train=True):
        for type_j,loss_j in loss_dict.items():

            model_i,params=get_model(data_i)
            loss_j(model_i,params)
            y=tf.one_hot(data_i.y,depth=params['n_cats'])
            history=model_i.fit(x=data_i.X,
        	                y=y,
                            epochs=n_epochs)
            history_dict[type_j].append( history.history )
    series_dict={key_i:defaultdict(lambda:[])
                  for key_i in history_dict}
    for key_i,hist_i in history_dict.items():
        for hist_j in hist_i:
            for name_k,metric_k in hist_j.items():
                series_dict[key_i][name_k].append(metric_k)
        for name_k,value_k in series_dict[key_i].items():
            arr=np.array(value_k)
            arr=np.mean(arr,axis=0)
            series_dict[key_i][name_k]=arr
    metrics=list(series_dict.values())[0].keys()
    t=np.arange(n_epochs)
    for metric_i in metrics:
        fig, ax = plt.subplots()
        for loss_type_j,dict_j in series_dict.items():
            series_j=dict_j[metric_i]
            ax.plot(t, series_j,label=loss_type_j)
            ax.set(title=metric_i)
        plt.xlabel("epochs")
        plt.ylabel(metric_i)
        plt.legend()
        plt.show()
        plt.clf()

def base_loss(model,params):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def custom_loss(model,params):
    loss=deep.weighted_loss(specific=None,
                       class_dict=params['class_weights'])
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def get_model(data):
    params={'dims': (data.dim(),),
            'n_cats':data.n_cats(),
            'n_epochs':100,
            'class_weights':dataset.get_class_weights(data.y)}
    hyper_params=ens.default_hyperparams()
    input_layer = Input(shape=(params['dims']))
    nn=deep.nn_builder(params=params,
                    hyper_params=hyper_params,
                    input_layer=input_layer,
                    i=0,
                    n_cats=params['n_cats'])
    model= Model(inputs=input_layer, 
                outputs=nn)
    return model,params

show_loss(data_path="../uci/wall-following",
	      split_path="single_exp/wall-following/splits")