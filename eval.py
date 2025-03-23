import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import dataset,pred,utils

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

def read_dynamic_subsets(in_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        result_path=f"{in_path}/purity_ens/results"
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

def eval_exp(conf_path):
    conf_dict=utils.read_json(conf_path)
    if(conf_dict["eval_type"]=="selection"):
        selection_eval(conf_dict)
    if(conf_dict["eval_type"]=="shapley"):
        shapley_eval(conf_dict)

def shapley_eval(conf_dict):
    subset_path=conf_dict["subset_path"]
    if(not os.path.isdir(subset_path)):
        utils.make_dir(subset_path)
        dynamic_subsets=read_dynamic_subsets(conf_dict["exp_path"])
        subset_iter=dynamic_subsets.all_subsets(conf_dict["metric_type"])
        for name_i,values_i in dynamic_subsets.all_subsets():
            print(name_i)
            utils.save_json(values_i,f"{subset_path}/{name_i}")
    subset_dict=read_static_subsets(subset_path)
    ord_dict=utils.read_json(conf_dict["ord_path"])
    point_dict={}
    for name_i,subset_i in subset_dict.items():
        n_clfs=subset_i.n_clfs()
        ord_i=ord_dict[name_i]
        if(len(ord_i) <n_clfs):
            n_clfs-=1
        shapley=[subset_i.shapley(k) for k in range(n_clfs)]
        point_dict[name_i]=(ord_i,shapley)
#    indiv_scatter(point_dict,conf_dict["plot_path"])
    total_scater(point_dict,conf_dict)

def indiv_scatter(point_dict,plot_path):
    plot_path=conf_dict["plot_path"]
    utils.make_dir(plot_path)
    for name_i,(x_i,y_i) in point_dict.items():
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_i, y_i)
        plt.savefig(f'{plot_path}/{name_i}')

def total_scater(point_dict,conf_dict):
    x,y=[],[]
    for name_i,(x_i,y_i) in point_dict.items():
        x.append(x_i)
        y.append(y_i)
    x,y=np.concatenate(x),np.concatenate(y)
    print(scipy.stats.pearsonr(x, y) )
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.title("Classes in each dataset")
    plt.ylabel(conf_dict["eval_type"])
    x_label=conf_dict["ord_path"].split("/")[-1].split(".")[0]
    plt.xlabel(y_label)
    plt.show()

def get_id(subset):
    id_j=list(subset)
    id_j.sort()
    return str(id_j)

def selection_eval(conf_dict):
    if(conf_dict["subplots"] is None):
        sig_df=pred.stat_test(exp_path=conf_dict["exp_path"],
                              clf_x="RF",
                              clf_y="class_ens",
                              metric_type=conf_dict["metric_type"])
        subplots=pred.sig_subsets(sig_df)
    else:
        subplots=conf_dict["subplots"]
    print(subplots)
    dynamic_subsets=read_dynamic_subsets(conf_dict["exp_path"])
    ord_dict=utils.read_json(conf_dict["ord_path"])

    def helper(name_i,subsets_i):
        ord_i=ord_dict[name_i]
        ord_i=np.argsort(ord_i)
        acc=subsets_i.order_acc(ord_i)         
        acc=np.array(acc)
        acc= np.mean(acc,axis=1)
#        if(z_score):
#            acc= acc-np.mean(acc)
        return acc
    acc_dict=dynamic_subsets.transform(helper)
    subplots={ key_i: [ (name_j,acc_dict[name_j])
               for name_j in value_i] 
                   for key_i,value_i in subplots.items()}
    make_plot(subplots)

def make_plot(all_subplots,
              title="Size",
              x_label="n_clf",
              y_label="acc",
              default_x=True):
    for i,(title_i,subplot_i) in enumerate(all_subplots.items()):
        _, ax_k = plt.subplots()
        print(subplot_i)
        for name_j,value_j in subplot_i:
            if(default_x):
                y=value_j
                x=np.arange(len(y))+1
            else:
                x,y=value_j
            ax_k.plot(x,y,
                      label=name_j)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        plt.clf()  

def sig_summary(exp_path):
    clf_types=['class_ens','deep','purity_ens','separ_class_ens','separ_purity_ens']
    metrics=['acc','balance']
    for metric_i in metrics:
        hist_i,data=None,None
        for j,clf_j in  enumerate(clf_types):
            df_ij=pred.stat_test(exp_path=exp_path,
                                 clf_x="RF",
                                 clf_y=clf_j,
                                 metric_type=metric_i)
            sig_dict_ij=sig_dict(df_ij,verbose=False)
            if(hist_i is None):
                data=df_ij['data']
                hist_i=np.zeros((len(data),len(clf_types)))
                data={data_k:k  for k,data_k in enumerate(data)}
            for k,key_k in enumerate(["worse","no_sig",'better']):
                for data_t in sig_dict_ij[key_k]: 
                    hist_i[data[data_t]][j]=k-1
        lines=[]
        for data_k,k in data.items():
            lines.append([data_k] +hist_i[k].tolist())
        sig_df=pd.DataFrame.from_records(lines,
                                         columns=["dataset"]+clf_types)
        print(sig_df)

def sig_dict(df,verbose=True):
    if(type(df)==str):
        df=pred.stat_test(exp_path=df,
                          clf_x="RF",
                          clf_y="class_ens",
                          metric_type="acc")
    if(verbose):
        print(df)    
    sig_dict={'no_sig':df['data'][df['sig']==False].tolist()}
    sig_df=df[df["sig"]==True]
    sig_dict['better']=sig_df['data'][ sig_df['diff']<0].tolist()
    sig_dict['worse']=sig_df['data'][ sig_df['diff']>0].tolist()
    return sig_dict

def find_best(in_path,nn_only=False):
    df=pred.summary(exp_path="new_exp")
    if(nn_only):
        df=df[df['clf']!='RF']
    dataset=df['data'].unique()
    id_acc=df_group=df.groupby('data')['acc'].idxmax()
    df_acc=df.loc[id_acc,]
    print(df_acc)
    id_balance=df_group=df.groupby('data')['balance'].idxmax()
    df_balance=df.loc[id_balance,]
    print(df_balance)

sig_summary("new_exp")

#history_acc("new_exp")
#eval_exp("new_exp",
#         ord_path="ord/size.json")
#eval_exp(conf_path="conf/basic2.js")
