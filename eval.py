import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from selection import compute_shapley
import dataset,pred,plot,selection,utils

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
    if(conf["type"]=="bar"):
        bar_plot(conf)
    if(conf["type"]=="box"):
        box_plot(conf)
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
            plot.scatter_plot(points=points_i,
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

def read_desc(desc_path):
    if(type(desc_path)==list):
        sub_dfs=[pd.read_csv(path_i) for path_i in desc_path]
        df= sub_dfs[0].merge(sub_dfs[1],on='dataset')#,axis=1)#'dataset')
        return df
    else:
        return pd.read_csv(desc_path)

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
        plot.scatter_plot(points,
                title=key_i,
                clf_x=conf['ord_value']['name'],
                clf_y=conf['metric'])

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
    plot.plot_series(series_dict,
                title=conf["title"],
                x_label=conf['x_feat'],
                y_label=conf['y_feat'],
                plt_limts=plt_limts)

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
    if(conf["var"]):
        def helper(pair_i):
            data_i,ens_variants_i=pair_i
            line_i=[data_i]
            for _,value_j in ens_variants_i:
                line_i.append(np.std(value_j))
            return line_i
        df=dataset.make_df(helper=helper,
                           iterable=value_dict.items(),
                           cols=['data']+conf["ens_types"])
        df.print()
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
        df.print()

def df_eval(conf):
    if('summary' in conf):
        s_conf=conf['summary']
        if(not 'selector' in s_conf):
            s_conf['selector']=None
        df=pred.summary(exp_path=conf['exp_path'],
                        selector=s_conf['selector'],
                        metrics=s_conf['metrics'])
        if(s_conf['sort']):
            df.group(s_conf['sort'])
        else:
            df.print()
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
                    metrics=s_conf['metrics'],
                    show=s_conf['plot'])

def sig_summary(exp_path,
                main_clf="RF",
                clf_types=None,
                metrics=None,
                show=False):
    if(clf_types is None):
        clf_types=['deep','class_ens','purity_ens',
                   'separ_class_ens','separ_purity_ens']
    if(metrics is None):
        metrics=['acc','balance']
    clf_types=[ type_i for type_i in clf_types
                    if(type_i!=main_clf)]
    def helper(metric_i):
        sig_matrix,data=[],None
        fun=lambda x: np.sign(x['diff'])*int(x['sig'])
        for j,clf_j in  enumerate(clf_types):
            df_ij=pred.stat_test(exp_path=exp_path,
                                 clf_x=main_clf,
                                 clf_y=clf_j,
                                 metric_type=metric_i)
            df_ij['sig_total']=df_ij.apply(fun, axis=1 )
            sig_matrix.append(df_ij['sig_total'].tolist())
            if(data is None):
                data=df_ij['data'].tolist()
        return np.array(sig_matrix),data        
    for metric_i in metrics:
        sig_matrix_i,data_i=helper(metric_i)
        def fun_i(tuple_j):
            j,data_j=tuple_j
            return [data_j]+sig_matrix_i[:,j].tolist()
        df_i=dataset.make_df(helper=fun_i,
                             iterable=enumerate(data_i),
                             cols=['data']+clf_types)
        df_i.print()
        if(show):
            plot.heatmap(matrix=sig_matrix_i.T,
                         x_labels=clf_types,
                         y_labels=data_i,
                         title=f"Statistical significance ({main_clf})")

#def sig_dict(df,verbose=True):
#    if(type(df)==str):
#        df=pred.stat_test(exp_path=df,
#                          clf_x="RF",
#                          clf_y="class_ens",
#                          metric_type="acc")
#    if(verbose):
#        print(df)    
#    sig_dict={'no_sig':df['data'][df['sig']==False].tolist()}
#    sig_df=df[df["sig"]==True]
#    sig_dict['better']=sig_df['data'][ sig_df['diff']<0].tolist()
#    sig_dict['worse']=sig_df['data'][ sig_df['diff']>0].tolist()
#    return sig_dict

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

def bar_plot(conf):
    df=pred.summary(exp_path=conf['exp_path'],
                    selector=conf['selector'],
                    metrics=[conf['metrics']])
    df.rename(col="clf",old="deep",new="MLP")
    df.print()
    acc_dict= df.as_dict(x_col='clf',y_col='acc')
    data=conf['data']
    clf_types=df.df['clf'].unique()
    step=len(clf_types)
    plot.bar_plot(acc_dict,data,clf_types,step)

def box_plot(conf):
    selector=pred.EnsSelector(words=conf['selector'],
                             necscf=conf['necscf'])
    @utils.EnsembleFun(in_path=('in_path',0),selector=selector)
    def helper(in_path):
        _,result=pred.get_result(in_path)
        return result.get_metric(conf['metric'])
    output=helper(conf['exp_path'])
    output=utils.rename_output(output,{"deep":"MLP"})
    data=set(conf['data'])
    values,clf_types=[],[]
    for data_i,clf_i,value_i in output:
        if( data_i in data):
            values.append(value_i)
            clf_types.append(clf_i)
    plot.box_plot(values=values,
                  names=list(data),
                  clf_types=clf_types)

if __name__ == '__main__':
#    sig_summary("new_exp")
#    find_best("new_exp")
    eval_exp("new_eval/conf/df.js")