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

    def is_fig(self):
        return (self.data_type=="fig")
    
    def save(self,out_path,verbose=True):
        if(self.is_fig()):
            self.data.savefig(out_path)
        else:
            with open(out_path, "w") as f:
                f.write(self.data.to_csv())
        if(verbose):
            print(f"Saved at {out_path}")

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
    return fun(conf)

def meta_eval(conf):
    if("taboo" in conf):
        fun_desc=[ fun_i for fun_i,tab_i in zip(conf["fun_used"],conf["taboo"])
                    if(tab_i)]
    else:
        fun_desc=conf["fun_used"]
    for conf_i,out_i,name_i in fun_desc:
        utils.make_dir(out_i)
        conf_i=read_conf(conf_i)
        fun_outputs=eval_exp(conf_i)
        if(type(name_i)!=list):
            name_i=[name_i]
        save_outputs(fun_outputs,name_i,out_i)

def save_outputs(fun_outputs,names,out_path):
    fig_index,df_index=0,0
    for fun_i in fun_outputs:
        if(fun_i.is_fig()):
            name_i=names[fig_index]
            fun_i.save(f"{out_path}/{name_i}.png")
            fig_index+=1
        else:
            name_i=names[df_index]
            fun_i.save(f"{out_path}/{name_i}.csv")
            df_index+=1

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
            if(s_conf['output']):
                out_i=f"{s_conf['output']}/{i}"
            else:
                out_i=None
            out_i=sig_summary(exp_path=conf['exp_path'],
                               main_clf=main_i,
                               clf_types=s_conf['clf_types'],
                               metric=metric_i,
                               show=False,
                               out_path=out_i)
            outputs+=out_i
        return outputs

def sig_summary(exp_path,
                main_clf="RF",
                clf_types=None,
                metric=None,
                show=False,
                out_path=None):
    print(main_clf,metric)
    clf_types=[ type_i for type_i in clf_types
                    if(type_i!=main_clf)]
    sig_matrix,data=[],None
    for i,clf_i in  enumerate(clf_types):
        df_i=pred.stat_test(exp_path=exp_path,
                             clf_x=main_clf,
                             clf_y=clf_i,
                             metric_type=metric)
        print(df_i)
        sig_dict_i={ dict_i['data']:dict_i['sig'] 
                        for dict_i in df_i.to_dict(orient='records')}
        if(data is None):
            data=list(sig_dict_i.keys())
            data.sort()
        sig_matrix.append([sig_dict_i[data_i] for data_i in data])    
    sig_matrix= np.array(sig_matrix) 
    def fun(tuple_i):
        i,data_i=tuple_i
        return [data_i]+sig_matrix[:,i].tolist()
    df=dataset.make_df(helper=fun,
                       iterable=enumerate(data),
                       cols=['data']+clf_types)
    output=[FunOuput("df",df)]
    clf_types=utils.rename(clf_types,old="deep",new='MLP')
    main_clf=utils.rename([main_clf],old="deep",new='MLP')[0]
    fig=plot.heatmap(matrix=sig_matrix.T,
                     x_labels=clf_types,
                     y_labels=data,
                     title=f"Statistical significance ({main_clf}/{metric})")
    output.append(FunOuput("fig",fig))
    return output

def box_plot(conf):
    selector=pred.EnsSelector(words=conf['selector'],
                             necscf=conf['necscf'])
    @utils.EnsembleFun(in_path=('in_path',0),selector=selector)
    def helper(in_path):
        print(in_path)
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
    def df_helper(tuple_i):
        data_i,dict_i=tuple_i
        lines=[]
        for clf_j,value_j in dict_i.items():
            line_j=[data_i,clf_i,np.mean(value_j),np.std(value_j)]
            lines.append(line_j)
        return lines
    df=dataset.make_df(helper=df_helper,
                       iterable=value_dict.items(),
                       cols=["clf","ens","acc","std"],
                       offset=None,
                       multi=True)
    outputs=[FunOuput("df",df)]
    clf_types=utils.rename(conf['selector'],old="deep",new='MLP')
    fig=plot.box_plot(value_dict=value_dict,
                      clf_types=clf_types,
                      show=conf['show'])
    outputs.append(FunOuput("fig",fig))
    return outputs

def xy_plot(conf):
    fig=plot.text_plot(x=utils.read_json(conf["x_plot"]),
                       y=utils.read_json(conf["y_plot"]),
                       title=conf["title"],
                       x_label=conf['x_label'],
                       y_label=conf['y_label'])
    return FunOuput("fig",fig)

def shapley_plot(conf):
    dict_x=get_series(conf["x"])
    def helper(y_i,title):
        dict_y=get_series(y_i)
        x,y=[],[]
        for key_i in dict_x:
            x+=dict_x[key_i]
            y+=dict_y[key_i]
        fig=plot.corl_plot(x,
                        y,
                        title=title,
                        clf_x=conf['x']['name'],
                        clf_y=y_i['name'])
        return FunOuput("fig",fig)
    return [ helper(y_i,f"{conf['title']} {y_i['ens_type']}") 
               for y_i in conf['y']]

def get_series(conf):
    if(conf["type"]=="shapley"):
        return compute_shapley(in_path=conf['subset_path'],
                               clf_type=conf['ens_type'],
                               metric_type=conf['metric'],
                               verbose=False)
    if(conf["type"]=="cls_desc"):
        return utils.read_json(conf["desc_path"])

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
    fig= plot.plot_series(series_dict,
                    title=conf["title"],
                    x_label=conf['x_feat'],
                    y_label=conf['y_feat'],
                    plt_limts=plt_limts)
    return FunOuput("fig",fig)

class SubsetDict(object):
    def __init__(self):
        self.by_id={}
        self.by_ens=defaultdict(lambda:[])
        self.by_data=defaultdict(lambda:[])
    
    def __len__(self):
        return len(self.by_id)

    def add(self,data,ens,value):
        current_id=len(self)+1
        self.by_id[current_id]=(data,ens,value)
        self.by_ens[ens].append(current_id)
        self.by_data[data].append(current_id)

    def transform(self,fun):
        self.by_id={ id_i:(data_i,ens_i,fun(data_i,ens_i,value_i))  
            for id_i,(data_i,ens_i,value_i) in self.by_id.items()} 
    
    def norm(self):
        def helper(data,ens,value):
            value*=100
            return np.round(value,2)
        self.transform(helper)

    def to_df(self):
        def helper(arg):
            data,ens,value=arg
            return [data,ens]+value.tolist()
        return dataset.make_df(helper=helper,
                       iterable=self.by_id.values(),
                       cols=['data','ens'],
                       offset="-")

    def data_dict(self,data):
        if(type(data)==list):
            return {data_i:self.data_dict(data_i)
                        for data_i in data}
        data_list=[self.by_id[i]
            for i in self.by_data[data]]
        return { ens_i:value_i for _,ens_i,value_i in data_list}
    
    def extr(self):
        ens_types=list(self.by_ens.keys())
        def helper(key):
            data_dict=self.data_dict(key)
            line_i=[key]
            for ens_i in ens_types:
                v_i=data_dict[ens_i]
                line_i.append(f"{min(v_i)}-{max(v_i)}%")
            return line_i
        cols=['data']+ens_types
        return dataset.make_df(helper=helper,
                       iterable=self.by_data.keys(),
                       cols=cols,
                       multi=False)

    def diff_dict(self):
        diff={}
        for data_i in self.by_data.keys():
            all_i=[]
            for value_j in self.data_dict(data_i).values():
                all_i+=value_j.tolist()
            diff[data_i]= max(all_i)-min(all_i)
        return diff


def make_subset_dict(subsets,metric_type):
    subset_dict=SubsetDict()
    for data_i in subsets[0].keys():
        for subset_j in subsets:
            ens_j,subset_j=subset_j[data_i]
            values_j=subset_j.mean_values(metric_type)
            values_j=np.round(values_j, 4)
            subset_dict.add(data_i,ens_j,values_j)
    return subset_dict

def subsets_plot(conf):
    subsets=[selection.subset_plot(conf["subset_path"],
                                   ens_type=ens_type_i)
                for ens_type_i in conf["ens_types"]]
    subset_dict=make_subset_dict(subsets,conf["metric"])
    if(conf["mlp_norm"]):
        def helper(data,ens,value):
            return value/conf["mlp"][data]
        subset_dict.transform(helper)
    else:
        subset_dict.transform(lambda d,e,v:v/v[-1])
    subset_dict.norm()
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
    df=subset_dict.extr()
#    df.clean("ens")
    df.print()
    diff_dict=subset_dict.diff_dict()
    diff_dict={ name_i:diff_i
                 for name_i,diff_i in diff_dict.items()
                    if(diff_i>0.035)}
    for data_i in conf["datasets"]:
        title = "MLP" if conf["mlp_norm"] else "full ensemble"
#        print(subset_dict.data_dict(data_i))
        plot.subset_plot(subset_dict.data_dict(data_i),
                     conf["ens_types"],
                     title=title)
    raise Exception(diff_dict)    
    output=[FunOuput("df",df)]
    plot.subset_plot(value_dict,
                     size_dict,
                     conf["ens_types"],
                     title=title)
    raise Exception(size_dict)
#    ens_dict={ens_i:{} for ens_i in conf["ens_types"]}
#    for data_i,list_i in value_dict.items():
#        for ens_j,value_j in list_i:
#            ens_dict[ens_j][data_i]=value_j        
#    for ens_i,dict_i in ens_dict.items():     
#        title = "MLP" if conf["mlp_norm"] else "full ensemble"
#        fig_i=plot.time_series(dict_i,
#                                title=f"{ens_i}/{title}",
#                                x_label='n_clfs',
#                                y_label='Accuracy')
#        output.append(FunOuput("fig",fig_i))
    return output


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



FUN_DICT={"meta":meta_eval,"selection":selection_plot,
          "desc":desc_plot,"subsets":subsets_plot,"bar":bar_plot,
          "box":box_plot,"df":df_eval,"shapley":shapley_plot,
          "xy_plot":xy_plot}

MULTI_FUN=set(["df","shapley","subsets"])

if __name__ == '__main__':
    outputs=eval_exp("uci_exp/conf/meta.js")
#    for i,out_i in enumerate(outputs):
#        out_i.save(f"{i}.png")