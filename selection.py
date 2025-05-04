import numpy as np
from tqdm import tqdm
import pandas as pd
import os.path
import ens,dataset,utils

class DynamicSubsets(object):
    def __init__(self,partial):
        self.partial=partial

    def all_subsets(self,metric_types="acc"):
        if(type(metric_types)==str):
            metric_types=[metric_types]
        n_clfs=self.partial.n_clfs()
        subsets=list(utils.powerset(range(n_clfs)))
        for subset_i in tqdm(subsets):
            value_i=[subset_i]
            for type_j in metric_types:
                metric_j=self.partial.get_metric(metric_type=type_j,
                                            subset=subset_i)
                metric_j=np.mean(metric_j)
                value_i.append(metric_j)
            yield value_i  

class StaticSubsets(object):
    def __init__(self,subset_dict,
                      value_dict,
                      metric_types):
        self.subset_dict=subset_dict
        self.value_dict=value_dict
        self.metric_types=metric_types
    
    def __call__(self,subset,metric_type='acc'):
        id_i=get_id(subset)
        k=self.metric_types[metric_type]
        return self.value_dict[id_i][k]

    def best(self,type):
        k=self.metric_types[type]
        best,best_subset=0,None
        for id_i,value_i in self.value_dict.items():
            value_k=value_i[k]
            if(best<value_k):
                best=value_k
                subset=id_i
        return best,subset

    def n_clfs(self):
        values=[len(subset_i) 
            for subset_i in self.subset_dict.values()]
        return max(values)

    def shapley(self,k,metric_type="acc"):
        singlton,margin=set([k]),[]
        m=self.metric_types[metric_type]
        for id_i,set_i in self.subset_dict.items():
            if(len(set_i)==1):
                continue
            if(k in set_i):
                diff_i=set_i.difference(singlton)
                one_out_id=get_id(diff_i)
                in_value=self.value_dict[id_i][m]
                out_value=self.value_dict[one_out_id][m]
                margin.append(in_value-out_value)
        return np.mean(margin)

    def mean_values(self,metric_type="acc"):
        metic_index=self.metric_types[metric_type]
        values=[[] for _ in range(self.n_clfs())]
        for id_i,tuple_i in self.subset_dict.items():
            size_i=len(tuple_i)-1
            metric_i=self.value_dict[id_i][metic_index]
            values[size_i].append(metric_i)
        values=[np.mean(value_i) for value_i in values]
        return np.array(values)

    def indv(self,metric_type="acc"):
        return [self([i],metric_type) for i in range(self.n_clfs())]

def read_static_subsets(in_path):
    raw=utils.read_json(in_path)
    metric_types,raw_subsets=raw['metric_types'],raw['subsets']
    metric_types={ type_i:i 
            for i,type_i in enumerate(metric_types)}
    subset_dict,value_dict={},{}
    for raw_i in raw_subsets:
        subset_i=raw_i[0]
        id_i=get_id(subset_i)
        values_i=raw_i[1:]
        subset_dict[id_i]=set(subset_i)
        value_dict[id_i]=values_i
    return StaticSubsets(subset_dict=subset_dict,
                         value_dict=value_dict,
                         metric_types=metric_types)

def get_id(subset):
    id_j=list(subset)
    id_j.sort()
    return str(id_j)

def subset_plot(subset_path,ens_type="class_ens"):
    @utils.EnsembleFun(selector=ens_type)
    def helper(in_path):
        return read_static_subsets(in_path)
    output_dict=helper(subset_path)
    return {data_i:(ens_i,subset_i) 
            for data_i,ens_i,subset_i in output_dict}

def best_df(in_path,glob=True):
    read=utils.EnsembleFun()(read_static_subsets)
    lines=[]
    for data_i,ens_i,value_i in  read(in_path):
        line_i=[data_i,ens_i]
        for type_k in value_i.metric_types:
            best_i,subset_i=value_i.best(type_k)
            line_i.append(best_i)
            line_i.append(subset_i.strip())
        lines.append(line_i)
    cols=["data","ens_type",'acc','acc_subset','balance','balance_subset']
    df=pd.DataFrame.from_records(lines,
                                  columns=cols)

    grouped=df.groupby(by="data")
    def helper(df):
        i= df["acc"].argmax()
        t= df["balance"].argmax()
        return df.iloc[[i,t]]
    df=grouped.apply(helper)
    print(df[["data","ens_type","acc","balance"]].round(4))
#    cols=["data","ens_type",'acc','acc_subset']
#    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
#        print(df[cols].round(4))#.to_latex())
    return df

def gen_subsets(in_path,
                out_path):
    @utils.EnsembleFun(out_path="out_path")
    def helper(in_path,out_path):
        if(os.path.exists(out_path)):
            return
        print(out_path)
        result_path=f"{in_path}/results"
        result_group=dataset.read_partial_group(result_path)
        dynamic_subsets=DynamicSubsets(result_group)
        metric_types=["acc","balance"]
        subsets_raw=list(dynamic_subsets.all_subsets(metric_types))
        output={"metric_types":metric_types, 
                "subsets":subsets_raw}
        utils.save_json(output,out_path)
    helper(in_path,out_path)

def compute_shapley(in_path,
                    clf_type="class_ens",
                    metric_type="balance",
                    verbose=False):
    @utils.EnsembleFun(selector=clf_type)
    def helper(in_path):
        subsets=read_static_subsets(in_path)
        return [subsets.shapley(k,metric_type=metric_type) 
                    for k in range(subsets.n_clfs())]
    output_list=helper(in_path)
    output_dict={}
    for data_i,_,value_i  in output_list:
        output_dict[data_i]=value_i
    if(verbose):
        print(output_dict)
    return output_dict

def shapley_stats(in_path,
                  clf_type="separ_purity_ens",
                  indiv=True):
    output_dict=compute_shapley(in_path,
                                clf_type=clf_type,
                                metric_type="acc")
    lines=[]
    if(indiv):
        for data_i,value_i in output_dict.items():
            for j,shapley_j in enumerate(value_i):
                lines.append([data_i,j,shapley_j])
        df=pd.DataFrame.from_records(lines,
                                  columns=["data","cat","shapley"])
        df=df.sort_values(by="shapley")
    else:
        for data_i,value_i in output_dict.items():
            lines.append([data_i,np.amax(value_i),np.mean(value_i),np.std(value_i)])
        df=pd.DataFrame.from_records(lines,
                                  columns=["data","max","mean","std"])
        df=df.sort_values(by="max")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.round(4))

if __name__ == '__main__':
#    gen_subsets("new_exp",
#                "new_eval/subsets")
    shapley_stats("new_eval/subsets")   