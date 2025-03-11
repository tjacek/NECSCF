import numpy as np
#import pandas as pd 
#from collections import namedtuple
#from collections import defaultdict
from tqdm import tqdm
import base,dataset,ens,deep,utils

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
        history=[]
        for model_i in self.clfs:
            history.append(model_i.fit(X,y))
        return history

    def predict(self,X):
        y=[clf_i.predict_proba(X) for clf_i in self.clfs]
        y=np.array(y)
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
    metric_dict,hist_dicts=eval_factory(data_split,clf_factory)
    print(metric_dict)

def eval_factory(data_split,clf_factory,metrics=None):
    if(metrics is None):
        metrics=["acc","balance"]
    def helper(split_i,clf_i):
        result_i,history_i=split_i.eval(data_split.data,
                                        clf_i)
        hist_dicts=[ utils.history_to_dict(history_j) 
                        for history_j in history_i]
        keys=hist_dicts[0].keys()
        single_dict={key_i:[hist_j[key_i] 
                             for hist_j in hist_dicts]
                        for key_i in keys}
        return result_i,single_dict
    output=[pair_i for pair_i in data_split.iter(helper,clf_factory)]
    results,history=list(zip(*output))
    history_stats={}
    for key_i in history[0].keys():
        if(not "loss" in key_i):
            raw_i=np.array([history_j[key_i] 
                        for history_j in history])
            history_stats[key_i]=np.mean(raw_i,axis=0)
    result=dataset.ResultGroup(results)
    metric_dict={ metric_i:np.mean(result.get_metric(metric_i))
                   for metric_i in metrics}
    return metric_dict,history

in_path="../uci/cleveland"
exp(in_path)