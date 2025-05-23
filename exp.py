def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import argparse,os,json
from collections import defaultdict
import base,dataset,deep,ens,pred,utils

def single_exp(in_path,
               out_path,
               ens_type="class_ens"):
    callback_type="total"
    utils.make_dir(out_path)
    data_split=get_splits(in_path,out_path)
    ens_path=f'{out_path}/{ens_type}'
    clf_factory=ens.get_custom_ens(callback_type=callback_type,
                                   verbose=1)
    utils.make_dir(ens_path)
    with open(f"{ens_path}/info.js", 'w') as f:
        json.dump({"ens":ens_type,"callback":callback_type}, f)
    model_path=f"{ens_path}/models"
    utils.make_dir(model_path)
    history_path=f"{ens_path}/history"
    utils.make_dir(history_path)
    for i,(clf_i,history_i) in enumerate(data_split.get_clfs(clf_factory)):
        print(type(clf_i))
        print(type(history_i))
        clf_i.save(f"{model_path}/{i}.keras")
        hist_dict_i=utils.history_to_dict(history_i)
        with open(f"{history_path}/{i}", 'w') as f:
            json.dump(hist_dict_i, f)

def eval_exp(data_path,exp_path="single_exp"):
    data_splits=get_splits(data_path,exp_path)
    for path_i in utils.top_files(exp_path):
        id_i=path_i.split("/")[-1]
        if(id_i!="splits"):
            partial_path=f"{path_i}/partial"
            info_dict=utils.read_json(f"{path_i}/info.js") 
            if(info_dict['ens']=='class_ens'):
                read=dataset.read_partial_group
                clf_type,partial="class_ens",True
            else:
                read=dataset.read_result_group
                clf_type,partial="MLP",False
            if(not os.path.isdir(partial_path)):
                print(f"Make {partial_path}")
                utils.make_dir(partial_path)
                make_results(data_splits=data_splits,
                                clf_factory=ens.get_ens(clf_type),
                                out_path=path_i,
                                result_path=partial_path,
                                partial=partial)                
            result=read(partial_path)
            print(id_i)
            print(f"Acc:{np.mean(result.get_metric(metric_type='acc')):.4f}")

def light_eval(data_path,
               exp_path="single_exp",
               ens_type="purity_ens"):
    data_splits=get_splits(data_path,exp_path)
    clf_factory=ens.get_ens(ens_type)
    ens_path=f"{exp_path}/{ens_type}"
    result_path=f"{ens_path}/results"
    if(not os.path.isdir(result_path)):
        utils.make_dir(result_path)
        for i,path_i in  enumerate(utils.top_files(f"{ens_path}/models")):
            model_path_i=f"{ens_path}/models/{i}.keras"
            print(model_path_i)
            clf_i=clf_factory.read(model_path_i)
            split_i=base.read_split(f"{exp_path}/splits/{i}.npz")#data_splits.splits[i]
            data_i= data_splits.data.selection(split_i.test_index)
            raw_i=clf_i.partial_predict(data_i.X)
            result_i=dataset.PartialResults(y_true=data_i.y,
                                        y_partial=raw_i)
            result_i.save(f"{result_path}/{i}.npz")
    results=dataset.read_partial_group(result_path)
    print(f"Acc:{np.mean(results.get_metric(metric_type='acc')):.4f}")
    print(f"Balance:{np.mean(results.get_metric(metric_type='balance')):.4f}")

def make_results(data_splits,
                    clf_factory,
                    out_path,
                    result_path,
                    partial=True):
    for i,data_i in data_splits.selection_iter(train=False):
        clf_i=clf_factory.read(f"{out_path}/models/{i}.keras")
        if(partial):
            raw_i=clf_i.partial_predict(data_i.X)
            result_i=dataset.PartialResults(y_true=data_i.y,
                                            y_partial=raw_i)
        else:
            raw_i=clf_i.predict(data_i.X)
            result_i=dataset.Result(y_true=data_i.y,
                                     y_pred=raw_i)
        result_i.save(f"{result_path}/{i}.npz")

def get_splits(in_path,out_path):
    split_path=f"{out_path}/splits"
    if(os.path.isdir(split_path)):
        return base.read_data_split(in_path,split_path)
    else:
        print("Make splits")
        data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
        utils.make_dir(split_path)
        for i,split_i in enumerate(data_split.splits):
            split_i.save(f"{split_path}/{i}")
        return data_split

def show_history(exp_path):
    for path_i in utils.top_files(exp_path):
        history_i=f"{path_i}/history"
        if os.path.exists(history_i):
            acc_dict=defaultdict(lambda:[])
            hist_dicts=[utils.read_json(path_j) 
                        for path_j in utils.top_files(history_i)]
            for key_i,value_i in hist_dicts[0].items():
                for hist_j in hist_dicts:
                    acc_dict[key_i].append(hist_j[key_i])
            print(path_i)
            for key_i,acc_i in acc_dict.items():
                if(not "loss" in key_i):
                    print(f"{key_i},{np.mean(acc_i):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../uci/cleveland")
    parser.add_argument("--output", type=str, default="new_exp/cleveland")
    parser.add_argument("--ens_type", type=str, default="purity_ens")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    if(args.train):
        single_exp(in_path=args.input,
               out_path=args.output,
               ens_type=args.ens_type)
    light_eval(data_path=args.input,
             exp_path=args.output)
    if(args.history):
        show_history(exp_path=args.output)