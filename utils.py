import os.path
import numpy as np
from functools import wraps
from itertools import chain, combinations
import multiprocessing
import re
import time,json

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths,key=natural_keys)
    return paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def read_json(in_path):
    with open(in_path, 'r') as file:
        data = json.load(file)
        return data

def save_json(value,out_path):
    with open(out_path, 'w') as f:
        json.dump(value, f)

class DirFun(object):
    def __init__(self,
                 dir_args=None,
                 input_arg='in_path',
                 out_arg='out_path'):
        if(dir_args is None):
            dir_args={"in_path":0}
        self.dir_args=dir_args
        self.input_arg=input_arg
        self.out_arg=out_arg

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            old_values=self.get_input(*args, **kwargs)
            in_path=old_values[self.input_arg]
            if(not os.path.isdir(in_path)):
                return fun(*args, **kwargs)
            if(self.out_arg in old_values):
                make_dir(old_values[self.out_arg])
            result_dict={}
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                new_values={name_j:f"{value_j}/{id_i}"  
                    for name_j,value_j in old_values.items()}
                result_dict[in_i]=self.eval_fun(fun,new_values,args,kwargs)
            return result_dict
        return decor_fun
    
    def get_input(self,*args, **kwargs):
        mod_values={}
        for arg,index in self.dir_args.items():
            if(arg in kwargs):
                mod_values[arg]=kwargs[arg]
            else:
                mod_values[arg]=args[index]
        return mod_values

    def eval_fun(self,fun,new_values,args,kwargs):
        args=list(args)
        for arg_i,i in self.dir_args.items():
            if(arg_i in kwargs):
                kwargs[arg_i]=new_values[arg_i]
            else:
                args[i]=new_values[arg_i]
        return fun(*args, **kwargs)

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
            original_args=FunArgs(args,kwargs)
            in_path=original_args.get(self.in_path)
            output=[]
            for path_i in top_files(in_path):
                id_i=path_i.split('/')[-1]
                if(self.out_path):
                    utils.make_dir(f"{self.out_path[0]}/{id_i}")
                for path_j in top_files(path_i):
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

class FunArgs(object):
    def __init__(self,args,kwargs):
        self.args=list(args)
        self.kwargs=kwargs
    
    def copy(self):
        return FunArgs(args=self.args.copy(),
                       kwargs=self.kwargs.copy())
    
    def get(self,pair):
        name,index=pair
        if(name in self.kwargs):
            return self.kwargs[name]
        else:
            return self.args[index]

    def set(self,pair,value):
        name,index=pair
        if(name in self.kwargs):
            self.kwargs[name]=value
        else:
            self.args[index]=value

class MultiDirFun(object):
    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            data_path=args[0]
            for path_i in top_files(data_path):
                id_i=path_i.split('/')[-1]
#                if(os.path.isdir(path_i)):
                new_args=[f"{arg_j}/{id_i}" for arg_j in args]
                p_i=multiprocessing.Process(target=fun, 
                                            args=new_args)
                p_i.start()
                p_i.join()
        return decor_fun
 
def elapsed_time(fun):
    @wraps(fun)
    def helper(*args, **kwargs):
        start=time.time()
        value=fun(*args, **kwargs)
        end= time.time()
        print(f"Time:{end-start:.4f}")
        return value
    return helper

def silence_warnings():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

def selected_subsets(order,full=False):
    order=list(order)
    subsets= [order[:i+1] for i in range(len(order))]
    if(full):
        size=len(subsets)
        subsets=[ subset_i+[size] for subset_i in subsets]
        subsets=[[size]]+subsets
    return subsets

def to_id_dir(path_dict,index=-1):
    return { path_i.split("/")[index] :value_i 
            for path_i,value_i in path_dict.items()
                if(value_i)}

def history_to_dict(history):
    if(type(history)==list):
        return [history_to_dict(history_i) for history_i in history]
    history=history.history
    key=list(history.keys())[0]
    hist_dict={'n_epochs':len(history[key])}
    for key_i in history.keys():
        hist_dict[key_i]=history[key_i][-1]
    return hist_dict

def mean_dict(all_dicts):
    mean_dict={ key_i:[]
        for key_i in all_dicts[0]}
    for key_i in mean_dict:
        raw_i=[dict_j[key_i] for dict_j in all_dicts]
        mean_dict[key_i]=(np.mean(raw_i),np.std(raw_i))
    return mean_dict

def extract_number(raw_str):
    return int("".join([str(d) for d in filter(str.isdigit, raw_str)]))

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)
                                                      if(n>0))
