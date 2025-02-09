import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
from tqdm import tqdm
#import gc,json
import os.path
import multiprocessing
import argparse
import pandas as pd
import base,dataset,ens,exp
from deep import weighted_loss
import utils,train

def pred_exp(data_path:str,
             exp_path:str,
             clf_type:str):
    
    if(clf_type=="class_ens"):
        fun=class_ens_pred
    if(clf_type=="RF"):
        fun=rf_pred
    helper=utils.MultiDirFun()(fun)
    output_dict=helper(data_path,exp_path)
    print(output_dict)

def class_ens_pred(in_path,exp_path):
    path_dir=train.get_paths(out_path=exp_path,
                            ens_type='class_ens',
                            dirs=['models','history','results'])
    try:
        chech_dirs(path_dir)
    except:
        return None
    data=dataset.read_csv(in_path)
    clf_factory=ens.get_ens("class_ens")
    pred_iter=model_iter(split_path=path_dir['splits'],
                         model_path=path_dir['models'],
                         ens_factory=clf_factory)
    utils.make_dir(path_dir["results"])
    for i,(split_i,clf_i) in tqdm(enumerate(pred_iter)):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=clf_i.partial_predict(test_data_i.X)
        result_i=dataset.PartialResults(y_true=test_data_i.y,
                                            y_partial=raw_partial_i)
        result_i.save(f"{path_dir['results']}/{i}.npz")

def rf_pred(in_path,exp_path):
    print(in_path)
    data=dataset.read_csv(in_path)
    clf_factory=base.ClfFactory(clf_type="RF")
    clf_factory.init(data)
    path_dir=train.get_paths(out_path=exp_path,
                                 ens_type='RF',
                                 dirs=['results','info.js'])

    utils.make_dir(path_dir['ens'])
    utils.make_dir(path_dir['results'])
    for j,split_j in tqdm(enumerate(split_iter(exp_path))):
        clf_j=clf_factory()
        result_j,_=split_j.eval(data,clf_j)
        result_j.save(f"{path_dir['results']}/{j}")
    with open(path_dict['info.js'], 'w') as f:
        json.dump({"ens":'RF',"callback":None}, f)

def chech_dirs(path_dir):
    if(not os.path.isdir(path_dir["models"])):
        raise Exception(f"Models don't exist")
#    if(os.path.isdir(path_dir["results"])):
#        raise Exception(f"Result exist")

def get_result(exp_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        result_path=f"{in_path}/class_ens/results"
        if(not os.path.isdir(result_path)):
             return None
        return [ dataset.read_partial(path_i)
                    for path_i in utils.top_files(result_path) ]
    result_dict=helper(exp_path)
    result_dict=utils.to_id_dir(result_dict,index=-1)
    metrics=["acc","balance"]
    lines=[]
    for name_i,results_i in result_dict.items():
        line_i=[name_i]
        for metric_j in metrics:
            value_j=[result_k.get_metric(metric_j)
                        for result_k in results_i]
            line_i.append(np.mean(value_j))
        lines.append(line_i)
    df=pd.DataFrame.from_records(lines,
                                  columns=["data"]+metrics)
    print(df.round(4))

def model_iter(split_path,model_path,ens_factory):
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


#def all_subsets(exp_path,subset_path):
#    result_dict=get_result(exp_path,
#                           acc=False)
#    utils.make_dir(subset_path)
#    for name_i,results_i in result_dict.items():
#        with open(f"{subset_path}/{name_i}.txt", 'w') as file:
#            cats=list(range(len(results_i[0])))
#            for subset_j in utils.powerset(cats):
#                acc_j=[result_k.selected_acc(subset_j) 
#                        for result_k in results_i]
#                mean_acc=np.mean(acc_j)
#                line_i=f"{subset_j}-{mean_acc:.4f}\n"
#                file.write(line_i)
#                print(line_i)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../uci")
    parser.add_argument("--exp_path", type=str, default="new_exp")
    parser.add_argument('--type', default=None, 
                         choices=[None,'class_ens','RF','MLP']) 
    args = parser.parse_args()
    if(args.type):
        pred_exp(data_path=args.data_path,
                 exp_path=args.exp_path,
                 clf_type=args.type)
    get_result(exp_path=args.exp_path)
    print(args)
