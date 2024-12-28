def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import os
import base,dataset,deep,ens,utils
import preproc

def single_exp(in_path,out_path):
    utils.make_dir(out_path)
    data_split=get_splits(in_path)
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data_split.data)
    result_group=data_split(clf_factory)
    result_group.save(f"{out_path}/class_ens")

def get_splits(in_path):
    split_path=f"{out_path}/splits"
    if(os.path.isdir(split_path)):
        return base.read_split(split_path)
    else:
        data_split=base.make_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
        utils.make_dir(split_path)
        for i,split_i in enumerate(data_split.splits):
            split_i.save(f"{split_path}/{i}")
        return data_split
       
#def selection(data):
#    sizes=data.class_percent()
#    return [ i for i,size_i in sizes.items()
#                  if(size_i<0.25) ]

#def selection_exp(in_path,
#                  n_splits=10,
#                  n_repeats=1):
#    data=dataset.read_csv(in_path)
#    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
#    splits=base.DataSplits( data=data,
#                            splits=protocol.get_split(data))
#    clf_factory=ens.ClassEnsFactory(selected_classes=None)
#    clfs=list(splits.get_clfs(clf_factory))
#    lines=[]
#    for subset_i in iter_subsets(data.n_cats()+1):
#        s_clfs=[SelectedEns(clf_j,subset_i) for clf_j in clfs]
#        results=splits.pred(s_clfs)
#        mean_i,balance_i=np.mean(results.get_acc()),np.mean(results.get_balanced())
#        lines.append([str(subset_i),mean_i,balance_i])
#    df=pd.DataFrame.from_records(lines,columns=['subset','acc','balance'])
#    return df   

#def iter_subsets(n_clfs):
#    cats= range(n_clfs)
#    for i in range(1,n_clfs):
#        cats_i=itertools.combinations(cats, i)
#        for cats_j in cats_i:
#            yield cats_j
#    yield list(cats)

if __name__ == '__main__':
    single_exp(in_path="../uci/vehicle",
               out_path="single_exp")