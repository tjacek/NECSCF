import numpy as np
import os
import matplotlib.pyplot as plt
import dataset,pred,utils

class DynamicSubsets(object):
    def __init__(self,partial_dict):
        self.partial_dict=partial_dict

    def transform(self,fun):
        return { name_i:fun(name_i,subsets_i)
                   for name_i,subsets_i in self.partial_dict.items()} 

def read_dynamic_subsets(in_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        result_path=f"{in_path}/class_ens/results"
        result_group=dataset.read_partial_group(result_path)
        return result_group
    output_dict=utils.to_id_dir(helper(in_path))
    return DynamicSubsets(output_dict)

def eval_exp(in_path,
             ord_path):
    dynamic_subsets=read_dynamic_subsets(in_path)
    ord_dict=utils.read_json(ord_path)
    def helper(name_i,subsets_i):
        ord_i=ord_dict[name_i]
        ord_i=np.argsort(ord_i)
        acc=subsets_i.order_acc(ord_i)         
        acc=np.array(acc)
        return np.mean(acc,axis=1)
    acc_dict=dynamic_subsets.transform(helper)
    print(acc_dict)
    sig_df=pred.stat_test(exp_path=in_path,
              clf_x="RF",
              clf_y="class_ens",
              metric_type="acc")
    subplots={}
    subplots["no_sig"]=list(sig_df[sig_df["sig"]==False]['data'])
    sig_df=sig_df[sig_df["sig"]==True]
    subplots["worse"]=list(sig_df[sig_df["diff"]>0]["data"])
    subplots["better"]=list(sig_df[sig_df["diff"]<0]["data"])
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

def history_acc(exp_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        for path_i in utils.top_files(in_path):
            history_i=f"{path_i}/history"
            if(os.path.exists(history_i)):
                hist=[utils.read_json(path_j) 
                            for path_j in utils.top_files(history_i)]
                keys=[key_j for key_j in hist[0].keys()
                        if(not "loss" in key_j)]
                hist_dict={ key_j:[hist_k[key_j] for hist_k in hist]
                             for key_j in keys}
                hist_dict={ key_j: (round(np.mean(value_j),4),
                	                round(np.std(value_j),4))
                             for key_j,value_j in hist_dict.items()}
                return hist_dict
    output_dict=helper(exp_path)
    for name_i,dict_i in output_dict.items():
    	print(name_i)
    	print(dict_i)

def sig_dict(df):
    if(type(df)==str):
        df=pred.stat_test(exp_path=df,
                          clf_x="RF",
                          clf_y="class_ens",
                          metric_type="acc")
    print(df)    
    sig_dict={'no_sig':df['data'][df['sig']==False].tolist()}
    sig_df=df[df["sig"]==True]
    sig_dict['better']=sig_df['data'][ sig_df['diff']<0].tolist()
    sig_dict['worse']=sig_df['data'][ sig_df['diff']>0].tolist()
    print(sig_dict)

#history_acc("new_exp")
eval_exp("new_exp",
         ord_path="ord/size.json") 