def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import argparse,os
import base,dataset,deep,ens,preproc,utils

def single_exp(in_path,
               out_path,
               ens_type="class_ens"):
    utils.make_dir(out_path)
    data_split=get_splits(in_path,out_path)
    ens_path=f'{out_path}/{ens_type}'
    clf_factory=ens.get_ens(ens_type)
#    clf_factory.init(data_split.data)
    utils.make_dir(ens_path)
    for clf_i,history_i in data_split.get_clfs(clf_factory):
        print(type(clf_i))
        print(type(history_i))
#def single_exp(in_path,
#               out_path,
#               ens_type="class_ens"):
#    utils.make_dir(out_path)
#    data_split=get_splits(in_path,out_path)
#    result_path=f"{out_path}/{ens_type}"
#    if(os.path.isdir(result_path)):
#        result_group=dataset.read_result_group(result_path)
#    else:
#        print(f"Train ens{ens_type}")
#        clf_factory=ens.get_ens(ens_type)
#        clf_factory.init(data_split.data)
#        result_group,history=data_split(clf_factory)
#        result_group.save(result_path)
#        history=utils.mean_dict(history)
#        print(history)
#    print(f"Acc:{result_group.get_acc()}")
#    print(f"Balance{result_group.get_acc()}")

def eval_exp(exp_path="single_exp"):
    for path_i in utils.top_files(exp_path):
        id_i=path_i.split("/")[-1]
        if(id_i!="splits"):
            results_i=dataset.read_result_group(path_i)
            print(id_i)
            print(f"Acc:{np.mean(results_i.get_acc())}")
            print(f"Balance{np.mean(results_i.get_balanced())}")            

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

def selection_exp(in_path,
                  out_path):
    utils.make_dir(out_path)
    data_split=get_splits(in_path,out_path)
    clf_factory=ens.get_ens("class_ens")()
    acc=[]
    for i,ens_i in enumerate(data_split.get_clfs(clf_factory)):
        split_i=data_split.splits[i]
        print(str(split_i))
        test_data=data_split.data.selection(split_i.test_index)
        y_partial=ens_i.partial_predict(test_data.X)
        y_partial=np.array(y_partial)
        result_i=dataset.PartialResults(y_true=test_data.y,
                                        y_partial=y_partial)
        acc.append(result_i.get_metric())
    acc=np.array(acc)
    print(np.mean(acc,axis=0))

def history_exp(in_path):
    data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
    clf_factory=ens.get_ens(ens_type="class_ens")
    clf_factory.init(data_split.data)
    clf=clf_factory()
    clf.verbose=2
    train_data=data_split.selection(i=0,train=True)   
    history=clf.fit(X=train_data.X,
                    y=train_data.y)
    hist_dict=utils.history_to_dict(history)
    for key_i,acc_i in hist_dict.items():
        print(f"{key_i}-{acc_i:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../uci/vehicle")
    parser.add_argument("--output", type=str, default="single_exp")
    parser.add_argument("--ens_type", type=str, default="class_ens")
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    if(args.eval):
        single_exp(in_path=args.input,
               out_path=args.output,
               ens_type=args.ens_type)
#        eval_exp(exp_path=args.output)
#    else:
#        history_exp(in_path=args.input)