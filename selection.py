import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset

class SubsetEval(object):
    def __init__(self,subsets,metrics):
        self.subsets=subsets
        self.metrics=metrics

    def compute_shapley(self,cat,metric_index=0):
        mean=np.mean(self.metrics[:,metric_index])
        
        cat_metric=[ self.metrics[i,metric_index]
                        for i,subset_i in enumerate(self.subsets)
                            if(cat in subset_i)]
        return np.mean(cat_metric)-mean

def get_subset(in_path):
    df=pd.read_csv(in_path)
    subsets,metrics=[],[]
    for i,row_i in df.iterrows():
        row_i=row_i.to_list()
        subsets.append(set(eval(row_i[1])))
        metrics.append(row_i[-2:])
    return SubsetEval(subsets=subsets,
    	              metrics=np.array(metrics))

def plot_shapley(data_path,subset_path="subset2.csv"):
    data=dataset.read_csv(data_path)
    sub_eval=get_subset(subset_path)
    shapley=[ sub_eval.compute_shapley(cat_i) 
                for cat_i in range(data.n_cats())]
    print(shapley)
    percent_dict=data.class_percent()
    plt.title(f"Classes in {data_path}")
    plt.scatter(x=[percent_dict[i]  for i in range(data.n_cats())], 
    	        y=shapley)
    plt.xlabel(f"Size")
    plt.ylabel("Shapley")
    plt.show()

plot_shapley("../uci/cleveland")