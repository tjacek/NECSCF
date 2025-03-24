import numpy as np
from functools import wraps
import ens,utils

def ensemble_fun(fun):
    @wraps(fun)
    def helper(in_path,out_path):
        if(out_path):
            utils.make_dir(out_path)
        for path_i in utils.top_files(in_path):
            id_i=path_i.split('/')[-1]
            for path_j in utils.top_files(path_i):
                id_j=path_j.split("/")[-1]
                if(ens.is_ensemble(id_j)):
                    out_ij=f"{out_path}/{id_i}/{id_j}"
                    fun(path_j,out_ij)
    return helper

class DynamicSubsets(object):
    def __init__(self,partial_dict):
        self.partial_dict=partial_dict

    def transform(self,fun):
        return { name_i:fun(name_i,subsets_i)
                   for name_i,subsets_i in self.partial_dict.items()}

    def all_subsets(self,metric_type="acc"):
        for name_i,partial_i in self.partial_dict.items():
            clf_i=partial_i.n_clfs()
            subsets_i=list(utils.powerset(range(clf_i)))
            values_i=[]
            for subset_j in tqdm(subsets_i):
                metric_j=partial_i.get_metric(metric_type=metric_type,
                                              subset=subset_j)
                metric_j=np.mean(metric_j)
                values_i.append((subset_j,metric_j))
            yield name_i,values_i  

def read_dynamic_subsets(in_path,clf_type):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        result_path=f"{in_path}/{clf_type}/results"
        result_group=dataset.read_partial_group(result_path)
        return result_group
    output_dict=utils.to_id_dir(helper(in_path))
    return DynamicSubsets(output_dict)

class StaticSubsets(object):
    def __init__(self,subset_dict,value_dict):
        self.subset_dict=subset_dict
        self.value_dict=value_dict
    
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

def read_static_subsets(in_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        raw_i=utils.read_json(in_path)
        subest_dict,value_dict={},{}
        for subset_j,value_j in raw_i:
            id_j=get_id(subset_j)
            subest_dict[id_j]=set(subset_j)
            value_dict[id_j]=value_j
        return StaticSubsets(subest_dict,value_dict)
    return utils.to_id_dir(helper(in_path))


def gen_subsets(in_path,out_path):
    @ensemble_fun
    def helper(in_path,out_path):
        print(in_path)
        print(out_path)
    helper(in_path,out_path)

gen_subsets("new_exp","subsets")