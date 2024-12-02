import numpy as np
import pandas as pd

class SubsetEval(object):
    def __init__(self,subsets,metrics):
        self.subsets=subsets
        self.metrics=metrics

    def compute_shapley(self,cat,metric_index=0):
        mean=np.mean(self.metrics[:,metric_index])
        
        cat_metric=[ self.metrics[i,metric_index]
                        for i,subset_i in enumerate(self.subsets)
                            if(cat in subset_i)]
        print(mean-np.mean(cat_metric))

def get_subset(in_path):
    df=pd.read_csv(in_path)
    subsets,metrics=[],[]
    for i,row_i in df.iterrows():
        row_i=row_i.to_list()
        subsets.append(set(eval(row_i[1])))
        metrics.append(row_i[-2:])
    return SubsetEval(subsets=subsets,
    	              metrics=np.array(metrics))

sub_eval=get_subset("subset.csv")
sub_eval.compute_shapley(4)