import numpy as np
import utils
utils.silence_warnings()
import argparse,os.path
import pandas as pd
from scipy import stats
from tqdm import tqdm
import base,dataset,ens

def pred_neural(data_path:str,
                 exp_path:str):
    @utils.MultiDirFun()
    def helper(in_path,exp_path):
        data=None
        for path_i in utils.top_files(exp_path):
            info_path=f"{path_i}/info.js" 
            if(not os.path.isfile(info_path)):
                continue
            info_dict=utils.read_json(info_path)
            if(not ens.is_neural(info_dict["ens"])):
                continue
            path_dir=base.get_paths(out_path=exp_path,
                                     ens_type=path_i.split("/")[-1],
                                     dirs=['models','results'])
            paths=get_paths(path_dir)
            if(len(paths)>0):
                clf_factory=ens.get_ens(info_dict["ens"])
                if(data is None):
                    data=dataset.read_csv(in_path)
                pred_from_models(data,
                                 paths,
                                 clf_factory)
    helper(data_path,exp_path)

def pred_clf(data_path:str,
                 exp_path:str,
                 clf_type="RF"):
    @utils.MultiDirFun()
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        path_dir=base.get_paths(out_path=exp_path,
                                 ens_type=clf_type,
                                 dirs=['results','info.js'])
        utils.make_dir(path_dir["ens"])
        clf_factory=ens.get_ens(clf_type)
        split_path=utils.top_files(path_dir['splits'])
        for i,split_path_i in tqdm(enumerate(split_path)):
            split_i=base.read_split(split_path_i)
            clf_i=clf_factory()
            split_i.fit_clf(data,clf_i)
            result_i=clf_i.eval(data,split_i)
            result_i.save(f"{path_dir['results']}/{i}.npz")
        utils.save_json(value=clf_factory.get_info(),
                        out_path=path_dict['info.js'])
    helper(data_path,exp_path)

def get_paths(path_dir):
    paths=[] 
    utils.make_dir(path_dir["results"])
    for i,model_path_i in enumerate(utils.top_files(path_dir["models"])):
        result_path_i=f"{path_dir['results']}/{i}.npz"
        if(not os.path.isfile(result_path_i)):
            split_path_i=f"{path_dir['splits']}/{i}.npz"
            paths.append((split_path_i,model_path_i,result_path_i))
    return paths

def pred_from_models(data,
                     paths,
                     clf_factory):
    for split_path_i,model_path_i,result_path_i in tqdm(paths):
        split_i=base.read_split(split_path_i)
        clf_i=clf_factory.read(model_path_i)
        result_i=clf_i.eval(data,split_i)
        result_i.save(result_path_i)

def get_result(path_i):
    info_dict=utils.read_json(f"{path_i}/info.js")
    clf_type=info_dict['ens']
    if("ens" in clf_type ):
        return clf_type,dataset.read_partial_group(f"{path_i}/results")
    else:
        return clf_type,dataset.read_result_group(f"{path_i}/results")

def summary(exp_path,
            selector=None,
            metrics=None,
            sort=False,
            std=False):
    if(selector is None):
        selector=basic_selector
    if(type(selector)==list):
        selector=EnsSelector(selector)
    if(metrics is None):
        metrics=['acc','balance']
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        output_dict=[]
        for path_i in utils.top_files(in_path):
            dir_id=path_i.split('/')[-1]
            if(selector(dir_id)):
                clf_type,result=get_result(path_i)
                output_dict.append((clf_type,result))
        return output_dict
    output_dict=helper(exp_path)
    result_dict=utils.to_id_dir(output_dict,index=-1)
    lines=[]
    for name_i,output_i in result_dict.items():
        for clf_j,result_j in output_i:
            line_j=[name_i,clf_j]
            for metric_k in metrics:
                value_j= result_j.get_metric(metric_k)
                line_j.append(np.mean(value_j))
                if(std):
                    line_j.append(np.std(value_j))
            lines.append(line_j)
    columns=["dataset","clf"]
    for metric_i in metrics:
        columns.append(metric_i)
        if(std):
            columns.append(f"{metric_i}_std")
    df=pd.DataFrame.from_records(lines,columns=columns)
    return dataset.DFView(df)

def basic_selector(dir_id):
    return  not "splits" in dir_id

class EnsSelector(object):
    def __init__(self,words):
        self.words=words

    def __call__(self,dir_id):
        if(not basic_selector(dir_id)):
            return False
        if("NECSCF" in dir_id):
            return False
        for word_i in self.words:
            if(word_i in dir_id):
                return True
        return False    

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../uci")
    parser.add_argument("--exp_path", type=str, default="new_exp")
    parser.add_argument('--nn', action='store_true')
    parser.add_argument('--clf',  type=str, default=None)
    parser.add_argument('--pairs', default='purity_ens,class_ens') 
    args = parser.parse_args()
    print(args)
    if(args.nn):
        pred_neural(data_path=args.data_path,
                    exp_path=args.exp_path)
    if(args.clf):
        pred_clf(data_path=args.data_path,
                 exp_path=args.exp_path)
    summary(exp_path=args.exp_path,
            metrics=["acc","balance"])
    if(args.pairs):
        clfs=args.pairs.split(',')
        if(len(clfs)>1):
            clf_x,clf_y=clfs[0],clfs[1]
            df_dict={}
            for metric_i in ["acc","balance"]:
                df_i=stat_test(exp_path=args.exp_path,
                               clf_x=clf_x,
                               clf_y=clf_y,
                                metric_type=metric_i)
                df_dict[metric_i]=df_i
            for name_i,df_i in df_dict.items():
            	print(name_i)
            	print(df_i)
        else:
            print(f"Not a pair:{args.pairs}")