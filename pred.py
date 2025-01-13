import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
from tqdm import tqdm
#import gc,json
import os.path
import multiprocessing
import argparse
import base,dataset,ens,exp
from deep import weighted_loss
import utils

def partial_exp(data_path:str,
                exp_path:str):
    @utils.MultiDirFun()
    def helper(in_path,exp_path):
        model_path=f"{exp_path}/class_ens"
        if(not os.path.isdir(model_path)):
            return None
        out_path=f"{exp_path}/partial"
        data=dataset.read_csv(in_path)
        ens_factory=ens.ClassEnsFactory()
        ens_factory.init(data)
        utils.make_dir(out_path)
        clf_iter=model_iter(exp_path,ens_factory)
        for i,(split_i,clf_i) in tqdm(enumerate(clf_iter)):
            test_data_i=data.selection(split_i.test_index)
            raw_partial_i=clf_i.partial_predict(test_data_i.X)
            result_i=dataset.PartialResults(y_true=test_data_i.y,
                                            y_partial=raw_partial_i)
            result_i.save(f"{out_path}/{i}.npz")
    helper(data_path,exp_path)

def get_result(exp_path,
               acc=False):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        partial_path=f"{in_path}/partial"
        if(not os.path.isdir(partial_path)):
            return None
        results=[dataset.read_partial(path_i) 
            for path_i in utils.top_files(partial_path) ]
        if(acc):
            return [result_i.get_metric("acc") for result_i in results]
        return results
    path_dict=helper(exp_path)
    return utils.to_id_dir(path_dict,index=-1)

def all_subsets(exp_path,subset_path):
    result_dict=get_result(exp_path,
                           acc=False)
    utils.make_dir(subset_path)
    for name_i,results_i in result_dict.items():
        with open(f"{subset_path}/{name_i}.csv", 'w') as file:
            cats=list(range(len(results_i[0])))
            for subset_j in utils.powerset(cats):
                acc_j=[result_k.selected_acc(subset_j) 
                        for result_k in results_i]
                mean_acc=np.mean(acc_j)
                line_i=f"{subset_j},{mean_acc:.4f}\n"
                file.write(line_i)
                print(line_i)

#def order_exp(data_path:str,
#              exp_path:str,
#              json_path:str,
#              out_path:str):
#    utils.make_dir(out_path)
#    exp_types={"base_full":(True,False),
#               "base":(False,False),
#               "reversed":(False,True),
#               "reversed_full":(True,True)}
#    for name_i,(full_i,reverse_i) in exp_types.items():
#        order_pred(data_path=data_path,
#                   exp_path=exp_path,
#                   json_path=json_path,
#                   out_path=f"{out_path}/{name_i}.json",
#                   full=full_i,
#                   reverse=reverse_i)

#def order_pred(data_path:str,
#               exp_path:str,
#               json_path:str,
#               out_path:str,
#               full=False,
#               reverse=False):
#    card_dict=utils.read_json(json_path)
#    def helper(data_path,exp_path,queue):
#        name=data_path.split("/")[-1]
#        print(name)
#        card= card_dict[name]
#        order=np.argsort(card)
#        if(reverse):
#            order=np.flip(order)
#        clf_selection=utils.selected_subsets(order.tolist(),
#                                             full=full)
#        data=dataset.read_csv(data_path)
#        ens_factory=ens.ClassEnsFactory()
#        ens_factory.init(data)
#        acc=[[] for _ in clf_selection]
#        for split_i,clf_i in tqdm(model_iter(exp_path,ens_factory)):
#            for j,subset_j in enumerate(clf_selection):
#                clf_j=exp.SelectedEns(clf_i,subset_j)
#                acc[j].append(split_i.pred(data,clf_j).get_acc())
#        acc=np.array(acc)
#        queue.put(np.mean(acc,axis=1).tolist())
#    acc_dict={}
#    for path_i in utils.top_files(data_path):
#        id_i=path_i.split("/")[-1]
#        exp_i=f"{exp_path}/{id_i}"
#        queue = multiprocessing.Queue()
#        p_i=multiprocessing.Process(target=helper, 
#                                    args=(path_i,exp_i,queue))
#        p_i.start()
#        p_i.join()
#        acc_dict[id_i]=queue.get()
#    with open(out_path, 'w', encoding='utf-8') as f:
#        json.dump(acc_dict, f, ensure_ascii=False, indent=4)
#    return acc_dict

#def result_pred(data_path,exp_path,out_path):
#    def helper(data_path,exp_path,out_path):
#        print(data_path)
#        data=dataset.read_csv(data_path)
#        ens_factory=ens.ClassEnsFactory()
#        ens_factory.init(data)
#        utils.make_dir(out_path)
#        for j,(split_j,clf_j) in tqdm(enumerate(model_iter(exp_path,ens_factory))):
#            result_j=split_j.pred(data,clf_j)
#            result_j.save(f"{out_path}/{j}")
#    utils.make_dir(out_path)
#    for path_i in utils.top_files(data_path):
#        id_i=path_i.split("/")[-1]
#        exp_i=f"{exp_path}/{id_i}"
#        out_i=f"{out_path}/{id_i}"
#        p_i=multiprocessing.Process(target=helper, 
#                                    args=(path_i,exp_i,out_i))
#        p_i.start()
#        p_i.join()

#def clf_pred(data_path,exp_path,out_path):
#    @utils.DirFun({"in_path":0,"exp_path":1,"out_path":2})
#    def helper(in_path,exp_path,out_path):
#        print(in_path)
#        data=dataset.read_csv(in_path)
#        clf_factory=base.ClfFactory(clf_type="RF")
#        clf_factory.init(data)
#        utils.make_dir(out_path)
#        for j,split_j in tqdm(enumerate(split_iter(exp_path))):
#            clf_j=clf_factory()
#            result_j=split_j.eval(data,clf_j)
#            result_j.save(f"{out_path}/{j}")
#        gc.collect()
#    helper(data_path,exp_path,out_path)


def model_iter(exp_path,ens_factory):
    split_path=f"{exp_path}/splits"
    model_path=f"{exp_path}/class_ens"
    for i,path_i in enumerate(utils.top_files(model_path)):
        split_i=f"{split_path}/{i}.npz"
        raw_split=np.load(split_i)
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        model_i=tf.keras.models.load_model(f"{model_path}/{i}.keras",
                                           custom_objects={"loss":weighted_loss})
        clf_i=ens_factory()
        clf_i.model=model_i
        yield split_i,clf_i

def split_iter(exp_path):
    split_path=f"{exp_path}/splits"
    for i,path_i in enumerate(utils.top_files(split_path)):
        split_i=f"{split_path}/{i}.npz"
        raw_split=np.load(split_i)
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        yield split_i

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../uci")
    parser.add_argument("--exp_path", type=str, default="exp_deep")
    parser.add_argument("--subset_path", type=str, default="subsets")
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()
    if(args.subset):
        all_subsets(exp_path=args.exp_path,
                    subset_path=args.subset_path)
    else:
        partial_exp(data_path=args.data_path,
                    exp_path=args.exp_path)