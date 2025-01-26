def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd
import argparse,os,json
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
            if(not os.path.isdir(partial_path)):
                print(f"Make {partial_path}")
                utils.make_dir(partial_path)
                clf_factory=ens.get_ens("class_ens")
                for j,split_j in enumerate(data_splits.splits):
                    test_data_j=data_splits.data.selection(split_j.test_index)
                    clf_j=clf_factory.read(f"{path_i}/models/{j}.keras")
                    raw_partial_j=clf_j.partial_predict(test_data_j.X)
                    result_j=dataset.PartialResults(y_true=test_data_j.y,
                                                    y_partial=raw_partial_j)
                    result_j.save(f"{partial_path}/{j}.npz")
            else:
                partial=dataset.read_partial_group(partial_path)
                print(id_i)
                print(f"Acc:{np.mean(partial.get_acc())}")
#            results_i=dataset.read_result_group(path_i)
#            print(id_i)
#            print(f"Acc:{np.mean(results_i.get_acc())}")
#            print(f"Balance{np.mean(results_i.get_balanced())}")            

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../uci/wall-following")
    parser.add_argument("--output", type=str, default="single_exp/wall-following")
    parser.add_argument("--ens_type", type=str, default="class_ens")
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    if(args.eval):
#        single_exp(in_path=args.input,
#               out_path=args.output,
#               ens_type=args.ens_type)
        eval_exp(data_path=args.input,
                 exp_path=args.output)
#    else:
#        history_exp(in_path=args.input)