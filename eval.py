import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from collections import defaultdict
from selection import compute_shapley
import dataset,pred,plot,selection,utils

class ConfDict(dict):
    def list_arg(self,arg='data'):
        if(not arg in self):
            return False
        return type(self[arg][0])!=str

    def iter(self,arg='data'):
        for data_i in self[arg]:
            conf_i= self.copy()
            conf_i[arg]=data_i
            yield ConfDict(conf_i)

    def product(self,x,y):
        for x_i in self[x]:
            for y_i in self[y]:
                yield x_i,y_i

    def get_dict(self,arg):
        return ConfDict(self[arg])

class FunOuput(object):
    def __init__(self,data_type,data):
        self.data_type=data_type
        self.data=data
    
    def save(self,out_path):
        self.data.savefig(out_path)

def read_conf(in_path):
    conf=utils.read_json(in_path)
    return ConfDict(conf)

def eval_exp(conf,show=False):
    if(type(conf)==str):
        conf=read_conf(conf)
    conf["show"]=show
    fun=FUN_DICT[conf['type']]
    if(conf.list_arg('data')):
        outputs=[]
        for conf_i in conf.iter('data'):
            outputs+= eval_exp(conf_i)
        return outputs
    if(conf['type']=="df"):
        return fun(conf)
    return [fun(conf)]
#    if(type(conf)==str):
#        conf=utils.read_json(conf)
#    if(conf['type']=="scatter"):
#        shapley_plot(conf)
#    if(conf['type']=='selection'):
#        selection_plot(conf)
#    if(conf['type']=="desc"):
#        desc_plot(conf)
#    if(conf['type']=="plot_xy"):
#        xy_plot(conf)
#    if(conf['type']=="subsets"):
#        subset_plot(conf)
#    if(conf["type"]=="bar"):
#        bar_plot(conf)
#    if(conf["type"]=="box"):
#        box_plot(conf)
#    if(conf['type']=='df'):
#        df_eval(conf)

def meta_eval(conf):
    for fun_id_i,(conf_i,out_i,name_i) in conf["fun_used"]:
        utils.make_dir(out_i)
        conf_i=read_conf(conf_i)

        fun_out=eval_exp(conf_i)
        for j,fun_j in enumerate(fun_out):
            fun_j.save(f"{out_i}/{name_i}{j}.png")

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
        s_conf=conf.get_dict('sig_summary')
        if(s_conf['output']):
            utils.make_dir(s_conf['output'])
        outputs=[]
        for i,(main_i,metric_i) in enumerate(s_conf.product("main_clf","metrics")):
            print(main_i,metric_i)
            if(s_conf['output']):
                out_i=f"{s_conf['output']}/{i}"
            else:
                out_i=None
            out_i=sig_summary(exp_path=conf['exp_path'],
                               main_clf=main_i,
                               clf_types=s_conf['clf_types'],
                               metric=metric_i,
                               show=s_conf['plot'],
                               out_path=out_i)
            outputs.append(out_i)
        return outputs

def sig_summary(exp_path,
                main_clf="RF",
                clf_types=None,
                metric=None,
                show=False,
                out_path=None):
    clf_types=[ type_i for type_i in clf_types
                    if(type_i!=main_clf)]
    sig_matrix,data=[],None
    fun=lambda x: np.sign(x['diff'])*int(x['sig'])
    for i,clf_i in  enumerate(clf_types):
        df_i=pred.stat_test(exp_path=exp_path,
                             clf_x=main_clf,
                             clf_y=clf_i,
                             metric_type=metric)
        df_i['sig_total']=df_i.apply(fun, axis=1 )
        sig_matrix.append(df_i['sig_total'].tolist())
        if(data is None):
            data=df_i['data'].tolist()
    sig_matrix= np.array(sig_matrix) 
    def fun(tuple_i):
        i,data_i=tuple_i
        return [data]+sig_matrix[:,i].tolist()
    df=dataset.make_df(helper=fun,
                       iterable=enumerate(data),
                       cols=['data']+clf_types)
#    print(df.to_csv())
#    if(show):
    clf_types=utils.rename(clf_types,old="deep",new='MLP')
    main_clf=utils.rename([main_clf],old="deep",new='MLP')[0]
    fig=plot.heatmap(matrix=sig_matrix.T,
                     x_labels=clf_types,
                     y_labels=data,
                     title=f"Statistical significance ({main_clf}/{metric})")
    fun= FunOuput("fig",fig)
    return fun

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
    print(df.to_csv())
    if(conf["plot"]):
        title = "(MLP)" if conf["mlp_norm"] else "(full ensemble)"
        plot.subset_plot(value_dict=value_dict,
                         data=conf['data'],
                         colors=['b','g','r','y'],
                         title=f"Clf selection {title}")

#def find_best(in_path,nn_only=False):
#    df=pred.summary(exp_path="new_exp")
#    if(nn_only):
#        df=df[df['clf']!='RF']
#    dataset=df['data'].unique()
#    id_acc=df_group=df.groupby('data')['acc'].idxmax()
#    df_acc=df.loc[id_acc,]
#    print(df_acc)
#    id_balance=df_group=df.groupby('data')['balance'].idxmax()
#    df_balance=df.loc[id_balance,]
#    print(df_balance)

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
        metric_value=result.get_metric(conf['metric'])
        if(conf["aggr"]):
            n_splits=conf["aggr"]
            n_iters=int(len(metric_value)/n_splits)
            index=[n_splits*i for i in range(n_iters)]
            return [np.mean(metric_value[i:(i+1)])  
                                for i in index]
        return metric_value 
    output=helper(conf['exp_path'])
    output=utils.rename_output(output,{"deep":"MLP"})
    value_dict=defaultdict(lambda :{})
    data=set(conf['data'])
    for data_i,clf_i,value_i in output:
        if( data_i in data):
            value_dict[data_i][clf_i]=value_i
    clf_types=utils.rename(conf['selector'],old="deep",new='MLP')
    fig=plot.box_plot(value_dict=value_dict,
                      clf_types=clf_types,
                      show=conf['show'])
    return FunOuput("fig",fig)

FUN_DICT={"meta":meta_eval,"selection":selection_plot,
          "desc":desc_plot,"subsets":subset_plot,"bar":bar_plot,
          "box":box_plot,"df":df_eval,"scatter":shapley_plot}

if __name__ == '__main__':
    eval_exp("conf/meta.js")
#    conf=read_conf("new_eval/conf/box.js")
#    if(conf.list_arg('data')):
#        for conf_i in conf.iter('data'):
#            eval_exp(conf_i)
#    else:
#        eval_exp(conf)