import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from functools import wraps
from selection import compute_shapley
import dataset,pred,utils

def eval_exp(conf):
    if(type(conf)==str):
        conf=utils.read_json(conf)
    if(conf['type']=="scatter"):
        shapley_plot(conf)
    if(conf['type']=="desc"):
        desc_plot(conf)

def shapley_plot(conf):
    if(type(conf)==str):
        conf=utils.read_json(conf)
    dict_x=get_series(conf["x"])
    dict_y=get_series(conf["y"])
    if(conf['plot']=='total'):    
        points=[]
        for key_i in dict_x:
            points_x,points_y=dict_x[key_i],dict_y[key_i]
            points+=list(zip(points_x,points_y))
        points=np.array(points)
        scatter_plot(points=points,
                     title=conf['title'],
                     clf_x=conf['x']['name'],
                     clf_y=conf['y']['name'])
    else:
        for key_i in dict_x: 
            points_x,points_y=dict_x[key_i],dict_y[key_i]
            points_i=np.array(list(zip(points_x,points_y)))
            scatter_plot(points=points_i,
                         title=key_i,
                         clf_x=conf['x']['name'],
                         clf_y=conf['y']['name'])

def get_series(param_dict):
    if(param_dict["type"]=="shapley"):
        return compute_shapley(in_path=param_dict['subset_path'],
                               clf_type=param_dict['name'],
                               metric_type=param_dict['metric'],
                               verbose=False)
    if(param_dict["type"]=="cls_desc"):
        return utils.read_json(param_dict["desc_path"])

def scatter_plot(points,
                 title,
                 clf_x, 
                 clf_y,
                 out_path=None):
    x,y=points[:,0], points[:,1]
    print(scipy.stats.pearsonr(x, y) )
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.title(title)
    plt.xlabel(clf_x)
    plt.ylabel(clf_y)
    plt.show()
    if(out_path):
        fig.savefig(f'{out_path}.png')

def desc_plot(conf):   
    desc_df=read_desc(conf['desc_path'])
    sig_df=pd.read_csv(conf['sig_path'])
    grouped=sig_df.groupby(by=conf["sig_clf"])
    def helper(df):
        return df['dataset'].tolist()
    out=grouped.apply(helper)
    series_dict={}
    for name_i,data_i in out.items():
        df_i=desc_df[desc_df['dataset'].isin(set(data_i))]
        data_names=df_i['dataset'].tolist()
        x=df_i[conf['x_feat']].tolist()
        y=df_i[conf['y_feat']].tolist()
        values=list(zip(x,y))
        points=list(zip(data_names,values))
        series_dict[name_i]=points
    x_feat=desc_df[conf['x_feat']]
    y_feat=desc_df[conf['y_feat']]
    plt_limts=((x_feat.min(),x_feat.max()+0.1),
               (y_feat.min(),y_feat.max()+0.1))
    plot_series(series_dict,
                title=conf["title"],
                x_label=conf['x_feat'],
                y_label=conf['y_feat'],
                plt_limts=plt_limts)

def read_desc(desc_path):
    if(type(desc_path)==list):
        sub_dfs=[pd.read_csv(path_i) for path_i in desc_path]
        df= sub_dfs[0].merge(sub_dfs[1],on='dataset')#,axis=1)#'dataset')
        return df
    else:
        return pd.read_csv(desc_path)

def plot_series(series_dict,
                title="Scatter",
                x_label='x',
                y_label='y',
                plt_limts=None):
    labels=['r','g','b']
    plt.figure()
    plt.title(title)
    for i,(_,points_i) in enumerate(series_dict.items()):
        for name_j,point_j in points_i:
            plt.text(point_j[0], 
                    point_j[1], 
                    name_j,
                    color=labels[i],
                    fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if(plt_limts):
        plt.xlim(plt_limts[0])
        plt.ylim(plt_limts[1])
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
    eval_exp("new_eval/conf/desc.js")
    #sig_summary("new_exp")
    #history_acc("new_exp")
    #eval_exp("new_exp",
    #         ord_path="ord/size.json")
    #eval_exp(conf_path="conf/basic2.js")
