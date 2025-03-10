import numpy as np
import pandas as pd 
from collections import namedtuple
from collections import defaultdict
import base,dataset,ens,deep

class FlexibleFactory(object):
    def __init__(self,weight_gen,hyper_params=None):
        if(hyper_params is None):
            hyper_params={'layers':2, 'units_0':2,
                          'units_1':1,'batch':False}
        self.hyper_params=hyper_params
        self.params=None
        self.weight_dict=None
        self.weight_gen=weight_gen

    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':500}
        self.weight_dict=dataset.get_class_weights(data.y)

    def __call__(self):
        clfs=[]
        for i in range(self.params["n_cats"]):
            params_i= self.params.copy()
            params_i['class_weights']=self.weight_gen(i=i,
                                                      weight_dict=self.weight_dict)
            clf_i=ens.Deep(params=params_i,
                       hyper_params=self.hyper_params)
            clfs.append(clf_i)
        return NaiveEnsemble(clfs)

class NaiveEnsemble(object):
    def __init__(self,clfs):	
        self.clfs=clfs

    def fit(self,X,y):
        for model_i in self.clfs:
            model_i.fit(X,y)

    def predict(self,X):
        y=[clf_i.predict_proba(X) for clf_i in self.clfs]
        y=np.array(y)
        print(y.shape)
        y=np.sum(y,axis=0)
        return np.argmax(y,axis=1)

def basic_weights(i,weight_dict):
    new_weight=weight_dict.copy()
    size=len(weight_dict)/2
    new_weight[i]*=size
    return new_weight

def exp(in_path):
    data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
    clf_factory=FlexibleFactory(weight_gen=basic_weights)
    result,_=data_split(clf_factory)
    acc=result.get_acc()
    print(acc)
    print(np.mean(acc))

in_path="../uci/cleveland"
exp(in_path)