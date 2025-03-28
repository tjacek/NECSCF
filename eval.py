import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from functools import wraps
from selection import compute_shapley
import dataset,pred,utils

def shapley_plot(conf):
    if(type(conf)==str):
        conf=utils.read_json(conf)
    dict_x=get_series(conf["x"])
    dict_y=get_series(conf["y"])
    points=[]
    for key_i in dict_x:
        points_x,points_y=dict_x[key_i],dict_y[key_i]
        points+=list(zip(points_x,points_y))
    points=np.array(points)
    scatter_plot(points,conf)

def get_series(param_dict):
    if(param_dict["type"]=="shapley"):
        return compute_shapley(in_path=param_dict['subset_path'],
                               clf_type=param_dict['name'],
                               metric_type=param_dict['metric'],
                               verbose=False)
    if(param_dict["type"]=="cls_desc"):
        return utils.read_json(param_dict["desc_path"])


def scatter_plot(points,conf):
    x,y=points[:,0], points[:,1]
    print(scipy.stats.pearsonr(x, y) )
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.xlabel(conf['x']['name'])
    plt.ylabel(conf['y']['name'])
    plt.show()


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

#def selection_eval(conf_dict):
#    if(conf_dict["subplots"] is None):
#        sig_df=pred.stat_test(exp_path=conf_dict["exp_path"],
#                              clf_x="RF",
#                              clf_y="class_ens",
#                              metric_type=conf_dict["metric_type"])
#        subplots=pred.sig_subsets(sig_df)
#    else:
#        subplots=conf_dict["subplots"]
#    print(subplots)
#    dynamic_subsets=read_dynamic_subsets(conf_dict["exp_path"])
#    ord_dict=utils.read_json(conf_dict["ord_path"])

#    def helper(name_i,subsets_i):
#        ord_i=ord_dict[name_i]
#        ord_i=np.argsort(ord_i)
#        acc=subsets_i.order_acc(ord_i)         
#        acc=np.array(acc)
#        acc= np.mean(acc,axis=1)
#        return acc
#    acc_dict=dynamic_subsets.transform(helper)
#    subplots={ key_i: [ (name_j,acc_dict[name_j])
#               for name_j in value_i] 
#                   for key_i,value_i in subplots.items()}
#    make_plot(subplots)

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
    clf_types=['deep','class_ens','purity_ens','separ_class_ens','separ_purity_ens']
    metrics=['acc','balance']
    for metric_i in metrics:
        hist_i,data=None,None
        for j,clf_j in  enumerate(clf_types):
            df_ij=pred.stat_test(exp_path=exp_path,
                                 clf_x="deep",
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
        print(sig_df.to_latex())

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


if __name__ == '__main__':
    shapley_plot("conf/basic2.js")
    #sig_summary("new_exp")
    #history_acc("new_exp")
    #eval_exp("new_exp",
    #         ord_path="ord/size.json")
    #eval_exp(conf_path="conf/basic2.js")
