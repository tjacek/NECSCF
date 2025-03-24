import numpy as np
from tqdm import tqdm
from functools import wraps
import ens,dataset,utils

def ensemble_fun(fun):
    @wraps(fun)
    def helper(in_path,out_path=None):
        if(out_path):
            utils.make_dir(out_path)
        output=[]
        for path_i in utils.top_files(in_path):
            id_i=path_i.split('/')[-1]
            if(out_path):
                utils.make_dir(f"{out_path}/{id_i}")
            for path_j in utils.top_files(path_i):
                id_j=path_j.split("/")[-1]
                if(ens.is_ensemble(id_j)):
                    out_ij=f"{out_path}/{id_i}/{id_j}"
                    value_ij=fun(path_j,out_ij)
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

#def read_dynamic_subsets(in_path,clf_type):
#    @utils.DirFun({"in_path":0})
#    def helper(in_path):
#        result_path=f"{in_path}/{clf_type}/results"
#        result_group=dataset.read_partial_group(result_path)
#        return result_group
#    output_dict=utils.to_id_dir(helper(in_path))
#    return DynamicSubsets(output_dict)

class StaticSubsets(object):
    def __init__(self,subset_dict,
                      value_dict,
                      metric_types):
        self.subset_dict=subset_dict
        self.value_dict=value_dict
        self.metric_types=metric_types
    
    def n_clfs(self):
        values=[len(subset_i) 
            for subset_i in self.subset_dict.values()]
        return max(values)

    def shapley(self,k):
        singlton,margin=set([k]),[]
        for id_i,set_i in self.subset_dict.items():
            if(len(set_i)==1):
                continue
            if(k in set_i):
                diff_i=set_i.difference(singlton)
                one_out_id=get_id(diff_i)
                in_value=self.value_dict[id_i]
                out_value=self.value_dict[one_out_id]
                margin.append(in_value-out_value)
        return np.mean(margin)

def read_static_subsets(in_path,out_path=None):
    print(in_path)
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
#def read_static_subsets(in_path):
#    @utils.DirFun({"in_path":0})
#    def helper(in_path):
#        raw_i=utils.read_json(in_path)
#        subest_dict,value_dict={},{}
#        for subset_j,value_j in raw_i:
#            id_j=get_id(subset_j)
#            subest_dict[id_j]=set(subset_j)
#            value_dict[id_j]=value_j
#        return StaticSubsets(subest_dict,value_dict)
#    return utils.to_id_dir(helper(in_path))


def max_acc(in_path):
    read=ensemble_fun(read_static_subsets)
    for data_i,ens_i,value_i in  read(in_path):
        print(data_i)
        print(ens_i)
        print(value_i)

def gen_subsets(in_path,out_path):
    @ensemble_fun
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

max_acc("subsets")