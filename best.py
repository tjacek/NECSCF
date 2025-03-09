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
        self.weight_dict=self.dataset.get_class_weights(data.y)

    def __call__(self):
        clfs=[]
        for i in range(self.params["n_cats"]):
            params_i= self.params.copy()
            params_i['class_weights']=self.weight_gen(i,params_i)
            clf_i=Deep(params=self.params,
                       hyper_params=self.hyper_params)
            clfs.appand(clf_i)
        return NaiveEnsemble(clfs)

class NaiveEnsemble(object):
    def __init__(self,clfs):	
        self.clfs=[]

    def fit(self,X,y):
        for model_i in self.clfs:
            model_i.fit(X,y)

def exp(in_path):
    data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
#    utils.make_dir(split_path)
    for i,split_i in enumerate(data_split.splits):
        print(split_i)

in_path="../uci/cleveland"
exp(in_path)