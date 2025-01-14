import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset,utils

class SubsetEval(object):
    def __init__(self,subsets,metrics):
        self.subsets=subsets
        self.metrics=metrics

    def n_clfs(self):
        return len(self.subsets[-1])
    
    def compute_shapley(self,cat,metric_index=0):
        mean=np.mean(self.metrics[:,metric_index])
        
        cat_metric=[ self.metrics[i,metric_index]
                        for i,subset_i in enumerate(self.subsets)
                            if(cat in subset_i)]
        return (np.mean(cat_metric)-mean)

    def by_size(self):
        subset_by_size=[[] for i in range(self.n_clfs()+1) ]
        for i,subset_i in enumerate(self.subsets):
            print(subset_i)
            subset_by_size[len(subset_i)].append(self.metrics[i])
        return subset_by_size

    def best(self):
        k=np.argmax(self.metrics[:,0])
        return self.metrics[k,0],self.subsets[k]

def get_subset(in_path):
    df=pd.read_csv(in_path)
    subsets,metrics=[],[]
    for i,row_i in df.iterrows():
        row_i=row_i.to_list()
        subsets.append(set(eval(row_i[1])))
        metrics.append(row_i[1:])
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

def size(subset_path="subsets"):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        n_clfs=[[] for _ in range(11)]
        with open(in_path, 'r') as file:
           for line_i in file:
               line_i=line_i.strip()
               tuple_i,acc_i=line_i.split("-")
               tuple_i,acc_i=eval(tuple_i),float(acc_i)
               print(len(tuple_i))
               n_clfs[len(tuple_i)-1].append(acc_i)
        return n_clfs
    size_dict=helper(subset_path)
    for name_i,clf_sizes in size_dict.items():
        size_i={j:np.mean(clf_j) 
                    for j,clf_j in enumerate(clf_sizes)
                       if(clf_j)}
        print(name_i)
        print(size_i)
#    sub_eval=get_subset(subset_path)
#    subsets= sub_eval.by_size()
#    for i,subsets_i in enumerate(subsets):
#        subsets_i= np.array(subsets_i)
#        print(np.mean(subsets_i,axis=0))   

if __name__ == '__main__':
    #plot_shapley("../uci/wine-quality-red")
    size(subset_path="subsets")