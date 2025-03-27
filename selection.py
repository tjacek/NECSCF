import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import wraps
import ens,dataset,utils

class EnsembleFun(object):
    def __init__(self,in_path=None,
                      out_path=None,
                      selector=None):
        if(in_path is None):
            in_path=("in_path",0)
        if(type(out_path)==str):
            out_path=(out_path,1)
        if(selector is None):
            selector=ens.is_ensemble
        if(type(selector)==str):
            clf_type=selector
            selector=lambda id_i: id_i==clf_type
        self.in_path=in_path
        self.out_path=out_path
        self.selector=selector

    def __call__(self,fun):
        @wraps(fun)
        def helper(*args, **kwargs):
            if(self.out_path):
                utils.make_dir(self.out_path[0])
            original_args=utils.FunArgs(args,kwargs)
            in_path=original_args.get(self.in_path)
            output=[]
            for path_i in utils.top_files(in_path):
                id_i=path_i.split('/')[-1]
                if(self.out_path):
                    utils.make_dir(f"{self.out_path[0]}/{id_i}")
                for path_j in utils.top_files(path_i):
                    id_j=path_j.split("/")[-1]
                    if(self.selector(id_j)):
                        new_args=original_args.copy()
                        new_args.set(self.in_path,path_j)
                        if(self.out_path):
                            new_args.set(self.out_path,f"{self.out_path[0]}/{id_i}/{id_j}")
                        value_ij=fun(*new_args.args,**new_args.kwargs)
                        output.append((id_i,id_j,value_ij))
            return output
        return helper

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

def read_static_subsets(in_path,out_path=None):
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

def best_df(in_path):
    read=ensemble_fun(read_static_subsets)
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
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        print(df[["data","ens_type",'acc','acc_subset']].round(4).to_latex())
    return df

def gen_subsets(in_path,out_path):
    @EnsembleFun("in_path","out_path")
    def helper(in_path,out_path):
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
    @EnsembleFun(selector=clf_type)
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

def shapley_plot(in_path):
    dict_x=compute_shapley(in_path,
                              clf_type="class_ens",
                              metric_type="acc",
                              verbose=False)
    dict_y=compute_shapley(in_path,
                           clf_type="separ_class_ens",
                              metric_type="acc",
                              verbose=False)
    points=[]
    for key_i in dict_x:
        points_x,points_y=dict_x[key_i],dict_y[key_i]
        points+=list(zip(points_x,points_y))
    points=np.array(points)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:,0], points[:,1])
    plt.ylabel("class_ens")
    plt.ylabel("separ_class_ens")
    plt.show()

shapley_plot("subsets")