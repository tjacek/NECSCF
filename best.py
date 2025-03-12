import numpy as np
#import pandas as pd 
from tqdm import tqdm
import base,dataset,desc,ens,deep,utils

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

class PurityWeights(object):
    def __init__(self,purity_hist):
        self.purity_hist=purity_hist

    def __call__(self,i,weight_dict):
        purity_i=self.purity_hist[i,:]
        new_weight={}
        for k,weight_k in weight_dict.items():
            if(k==i):
                new_weight[k]= 2*weight_k
            else:
                new_weight[k]= (1.0-purity_i[k])*weight_k
        return new_weight

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

def find_best(in_path,out_path):
    datasets=["cmc","vehicle"]
    utils.make_dir(out_path)
    for data_i in datasets:
        output_i=single_exp(f"{in_path}/{data_i}",verbose=True)
        utils.save_json(value=output_i,
                        out_path=f"{out_path}/{data_i}")

def single_exp(in_path,verbose=False):
    data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
    gen_dict={"purity":PurityWeights(desc.purity_hist(data_split.data)),
              "basic":basic_weights}
    output_dict={}
    for type_i,gen_i in gen_dict.items():
        clf_factory_i=FlexibleFactory(weight_gen=gen_i)
        metric_dict,history_stats=eval_factory(data_split,clf_factory_i)
        output_dict[type_i]=(metric_dict,history_stats)
        if(verbose):
             print(type_i)
             print(history_stats)
             print(metric_dict)
    return output_dict

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
            history_stats[key_i]=list(np.mean(raw_i,axis=0))
    result=dataset.ResultGroup(results)
    metric_dict={ metric_i:np.mean(result.get_metric(metric_i))
                   for metric_i in metrics}
    return metric_dict,history_stats

in_path="../uci/cleveland"
#exp(in_path)
find_best(in_path="../uci",
          out_path="best")