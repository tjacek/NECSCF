import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from selection import compute_shapley
import dataset,pred,selection,utils

def eval_exp(conf):
    if(type(conf)==str):
        conf=utils.read_json(conf)
    if(conf['type']=="scatter"):
        shapley_plot(conf)
    if(conf['type']=='selection'):
        selection_plot(conf)
    if(conf['type']=="desc"):
        desc_plot(conf)
    if(conf['type']=="plot_xy"):
        xy_plot(conf)
    if(conf['type']=="subsets"):
        subset_plot(conf)
    if(conf['type']=='df'):
        df_eval(conf)

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
                               clf_type=param_dict['ens_type'],
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
    pearson=scipy.stats.pearsonr(x, y) 
    text=f"corelation:{pearson.correlation:.4f},pvalue:{pearson.pvalue:.4f}"
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y)
    plt.title(title)
    plt.xlabel(clf_x)
    plt.ylabel(clf_y)
    ax.annotate(text,
                xy = (0.7, -0.15),
                xycoords='axes fraction',
                ha='right',
                va="center")
    fig.tight_layout()
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

def xy_plot(conf):   
    x=utils.read_json(conf["x_plot"])
    y=utils.read_json(conf["y_plot"])
    plt.figure()
    for key_i in x:
        x_i,y_i=x[key_i],y[key_i]
        plt.text(x_i, 
                 y_i, 
                 key_i,
                 fontdict={'weight': 'bold', 'size': 9})
    x_values=list(x.values())
    y_values=list(y.values())
    plt.title(conf["title"])
    plt.xlabel(conf['x_label'])
    plt.ylabel(conf['y_label'])
    plt.xlim((min(x_values),max(x_values)*1.25))
    plt.ylim((min(y_values),max(y_values)*1.25))
#    plt.ylim((min(x_values),max(x_values)*1.25))
    plt.show()

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

def selection_plot(conf):
    @utils.EnsembleFun(selector=conf["clf_type"])
    def helper(in_path):
        return selection.read_static_subsets(in_path)
    output= helper(conf["subset_path"])
    output_dict={key_i:value_i for key_i,_,value_i in output}
    ord_value=get_series(conf['ord_value'])
    for key_i in ord_value:
        value_i,all_subsets_i=ord_value[key_i],output_dict[key_i]
        order_i=np.argsort(value_i)
        metric_value=[all_subsets_i(subset_k,metric_type=conf['metric']) 
                        for subset_k in utils.selected_subsets(order_i)]
        x=[i for i in range(len(metric_value))]
        points=np.array([x,metric_value]).T
        scatter_plot(points,
                title=key_i,
                clf_x=conf['ord_value']['name'],
                clf_y=conf['metric'])

def subset_plot(conf):
    subsets=[selection.subset_plot(conf["subset_path"],
                                   ens_type=ens_type_i)
                for ens_type_i in conf["ens_types"]]
    value_dict={data_i:[] for data_i in subsets[0].keys()}
    for data_i in value_dict.keys():
        for subset_j in subsets:
            ens_j,subset_j=subset_j[data_i]
            values_j=subset_j.mean_values(conf["metric"])
            if(conf["mlp_norm"]):
                full_i=conf["mlp"][data_i]
                values_j*=100
            else:
                full_i=values_j[-1]
            values_j/=full_i
            values_j=np.round(values_j, 4)
            value_dict[data_i].append((ens_j,values_j))

    if(conf["plot"]):
        for data_i,pairs_i in value_dict.items():
            n_clf=pairs_i[0][1].shape[0]
            x_i=np.arange(n_clf)
            for ens_j,values_j in pairs_i:
                plt.plot(x_i, values_j, label = ens_j)
            plt.title(data_i)
            plt.legend()
            plt.show()
            plt.clf()
    else:
        def iterator():
            for data_i, ens_i in value_dict.items():
                for ens_j in ens_i:
                    yield data_i,ens_j
        def helper(arg_i):
            data_i,value_i=arg_i
            ens_i,arr_i=value_i
            arr_i*=100
            return [data_i,ens_i]+arr_i.tolist()
        df=dataset.make_df(helper=helper,
                           iterable=iterator(),
                           cols=['data','ens'],
                           offset="-")
        df.clean("ens")

        print(df.to_latex(no_empty=6))
#print(f"{name_i},{ens_j}")
#print(values_j)
#    indv_i=subsets_i.indv()
#    print(f"{np.mean(indv_i):.4f}:{np.std(indv_i):.4f}")

def df_eval(conf):
    if('summary' in conf):
        s_conf=conf['summary']
        if(not 'selector' in s_conf):
            s_conf['selector']=None
        pred.summary(exp_path=conf['exp_path'],
                     selector=s_conf['selector'],
                     metrics=s_conf['metrics'],
                     sort=s_conf['sort'])
    if('sig_pairs' in conf):
        s_conf=conf['sig_pairs']
        for pair_i in s_conf['pairs']:
            clf_x,clf_y=pair_i.split(",")
            df=pred.stat_test(conf['exp_path'],
                              clf_x,
                              clf_y,
                              metric_type=s_conf['metric'])
            print(df.round(4))
    if('sig_summary' in conf):
        s_conf=conf['sig_summary']
        sig_summary(exp_path=conf['exp_path'],
                    main_clf=s_conf["main_clf"],
                    clf_types=s_conf['clf_types'],
                    metrics=s_conf['metrics'])

def sig_summary(exp_path,
                main_clf="RF",
                clf_types=None,
                metrics=None):
    if(clf_types is None):
        clf_types=['deep','class_ens','purity_ens',
                   'separ_class_ens','separ_purity_ens']
    if(metrics is None):
        metrics=['acc','balance']
    for metric_i in metrics:
        hist_i,data=None,None
        for j,clf_j in  enumerate(clf_types):
            df_ij=pred.stat_test(exp_path=exp_path,
                                 clf_x=main_clf,
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
        sig_df=DFView(sig_df)
        sig_df.print()#.to_latex())

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
#    sig_summary("new_exp")
#    find_best("new_exp")
    eval_exp("new_eval/conf/subset.js")