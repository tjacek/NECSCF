import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os.path,json
from scipy import stats
#import multiprocessing
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
    path_dict=train.get_paths(out_path=exp_path,
                                 ens_type='RF',
                                 dirs=['results','info.js'])

    utils.make_dir(path_dict['ens'])
    utils.make_dir(path_dict['results'])
    for j,split_j in tqdm(enumerate(split_iter(exp_path))):
        clf_j=clf_factory()
        result_j,_=split_j.eval(data,clf_j)
        result_j.save(f"{path_dict['results']}/{j}")
    with open(path_dict['info.js'], 'w') as f:
        json.dump({"ens":'RF',"callback":None}, f)

def chech_dirs(path_dir):
    if(not os.path.isdir(path_dir["models"])):
        raise Exception(f"Models don't exist")
#    if(os.path.isdir(path_dir["results"])):
#        raise Exception(f"Result exist")

def summary(exp_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        print(in_path)
        output_dict=[]
        for path_i in utils.top_files(in_path):
            if(not "splits" in path_i):
                clf_type,result=get_result(path_i)
                output_dict.append((clf_type,result))
        return output_dict
    output_dict=helper(exp_path)
    result_dict=utils.to_id_dir(output_dict,index=-1)
    metrics=["acc","balance"]
    lines=[]
    for name_i,output_i in result_dict.items():
        for clf_j,result_j in output_i:
            line_j=[name_i,clf_j]
            for metric_k in metrics:
                value_j= result_j.get_metric(metric_k)
                line_j.append(np.mean(value_j))
            lines.append(line_j)
    df=pd.DataFrame.from_records(lines,
                                  columns=["data","clf"]+metrics)
    print(df.round(4))

def stat_test(exp_path,
              clf_x,
              clf_y,
              metric_type="acc"):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        _,result_x=get_result(f"{in_path}/{clf_x}")
        _,result_y=get_result(f"{in_path}/{clf_y}")
        x_value=result_x.get_metric(metric_type)
        y_value=result_y.get_metric(metric_type)
        mean_x,mean_y=np.mean(x_value),np.mean(y_value)
        diff= mean_x-mean_y
        pvalue=stats.ttest_ind(x_value,y_value,
                               equal_var=False)[1]
        return mean_x,mean_y,diff,round(pvalue,6)
    pvalue_dict=helper(exp_path)
    pvalue_dict=utils.to_id_dir(pvalue_dict,index=-1)
    lines=[ [name_i]+list(line_i) for name_i,line_i in pvalue_dict.items()]
    df=pd.DataFrame.from_records(lines,
                              columns=['data',clf_x,clf_y,"diff","pvalue"])
    df['sig']=df['pvalue'].apply(lambda pvalue_i:pvalue_i<0.05)
    df=df.sort_values(by='diff')
    return df

def sig_subsets(sig_df):
    subplots={}
    subplots["no_sig"]=list(sig_df[sig_df["sig"]==False]['data'])
    sig_df=sig_df[sig_df["sig"]==True]
    subplots["worse"]=list(sig_df[sig_df["diff"]>0]["data"])
    subplots["better"]=list(sig_df[sig_df["diff"]<0]["data"])
    return subplots

def get_result(path_i):
    info_dict=utils.read_json(f"{path_i}/info.js")
    clf_type=info_dict['ens']
    if(clf_type=="class_ens"):
        return clf_type,dataset.read_partial_group(f"{path_i}/results")
    else:
        return clf_type,dataset.read_result_group(f"{path_i}/results")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.aaropietrzakdd_argument("--data_path", type=str, default="../uci")
    parser.add_argument("--exp_path", type=str, default="new_exp")
    parser.add_argument('--type', default=None, 
                         choices=[None,'class_ens','RF','MLP']) 
    parser.add_argument('--pairs', default=None,) 
    args = parser.parse_args()
    if(args.type):
        pred_exp(data_path=args.data_path,
                 exp_path=args.exp_path,
                 clf_type=args.type)
#    summary(exp_path=args.exp_path)
    if(args.pairs):
        clfs=args.pairs.split(',')
        if(len(clfs)>1):
            clf_x,clf_y=clfs[0],clfs[1]
            stat_test(exp_path=args.exp_path,
                      clf_x=clf_x,
                      clf_y=clf_y)
        else:
            print(f"Not a pair{args.pairs}")