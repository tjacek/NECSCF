def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import base,dataset,deep,ens,utils

class SelectedEns(object):
    def __init__(self,ens,select_cats):
        self.ens=ens
        self.select_cats=select_cats

    def predict(self,X):
        return self.ens.select_predict(X=X,
                                       select_cats=self.select_cats)

def clf_exp(in_path,
            n_splits=10,
            n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=base.DataSplits( data=data,
                            splits=protocol.get_split(data))
     
#    selected_classes=selection(data)#[0,1,2]
    clfs={'RF':base.ClfFactory('RF'),
          'deep':ens.DeepFactory(),
          'class_ens':ens.ClassEnsFactory()}#selected_classes=selected_classes)}
    acc_dict,balance_dict={},{}
    for clf_type_i,clf_i in clfs.items():
        print(clf_type_i)
        results=splits(clf_i)
        acc_dict[clf_type_i]=np.mean(results.get_acc())
        balance_dict[clf_type_i]=np.mean(results.get_balanced() )
    print(acc_dict)
    print(balance_dict)

def selection(data):
    sizes=data.class_percent()
    return [ i for i,size_i in sizes.items()
                  if(size_i<0.25) ]

def selection_exp(in_path,
                  n_splits=10,
                  n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=base.DataSplits( data=data,
                            splits=protocol.get_split(data))
    clf_factory=ens.ClassEnsFactory(selected_classes=None)
    clfs=list(splits.get_clfs(clf_factory))
    lines=[]
    for subset_i in iter_subsets(data.n_cats()+1):
        s_clfs=[SelectedEns(clf_j,subset_i) for clf_j in clfs]
        results=splits.pred(s_clfs)
        mean_i,balance_i=np.mean(results.get_acc()),np.mean(results.get_balanced())
        lines.append([str(subset_i),mean_i,balance_i])
        print(lines[-1])
    df=pd.DataFrame.from_records(lines,columns=['subset','acc','balance'])
    return df   

def iter_subsets(n_clfs):
    cats= range(n_clfs)
    for i in range(1,n_clfs):
        cats_i=itertools.combinations(cats, i)
        for cats_j in cats_i:
            yield cats_j
    yield list(cats)

selection_exp(in_path="../uci/wine-quality-red")
#df.to_csv('subset2.csv')
