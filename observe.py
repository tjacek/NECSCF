import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from keras import Input, Model
from collections import defaultdict
import matplotlib.pyplot as plt
import base,dataset,deep,ens

def show_loss(data_path,
              split_path,
              n_epochs=40):
    data_split=base.read_data_split(data_path=data_path,
		                            split_path=split_path)
    loss_dict={'base':TrainAlg,
               'weight':WeightedLoss,
               'custom':CustomLoss}
    history_dict={key_i:[] for key_i in loss_dict}
    for i,data_i in data_split.selection_iter(train=True):
        for type_j,alg_type_j in loss_dict.items():
            model_i,params=get_model(data_i)
            loss_j=alg_type_j(model_i,params)
            loss_j.prepare_model()
            history=loss_j.fit(data=data_i,
                               n_epochs=n_epochs)
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

class TrainAlg(object):
    def __init__(self,model,params):
        self.model=model
        self.params=params

    def prepare_model(self):
        self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
        return self

    def fit(self,data,n_epochs):
        y=tf.one_hot(data.y,
                     depth=self.params['n_cats'])
        history=self.model.fit(x=data.X,
                               y=y,
                               epochs=n_epochs)
        return history

class CustomLoss(TrainAlg):
    def prepare_model(self):
        params=self.params['class_weights']
        loss=deep.weighted_loss(specific=None,
                       class_dict=params)
        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=['accuracy'],
                           jit_compile=False)
        return self

class WeightedLoss(TrainAlg):
    def fit(self,data,n_epochs):
        y=tf.one_hot(data.y,
                     depth=self.params['n_cats'])
        history=self.model.fit(x=data.X,
                               y=y,
                               epochs=n_epochs,
                               class_weight=self.params['class_weights'])
        return history

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

def check_f1_score(in_path):
    result=dataset.read_result_group(in_path)
    result=result.results[0]
    result.report()
    f1=result.get_metric("f1-score")
    print(f"{f1:.4f}") 
if __name__ == '__main__':
    check_f1_score("new_exp/wine-quality-red/deep/results")
#    show_loss(data_path="../uci/wall-following",
#	      split_path="single_exp/wall-following/splits")