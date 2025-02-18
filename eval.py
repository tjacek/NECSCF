import numpy as np
import os
import dataset,pred,utils

class DynamicSubsets(object):
    def __init__(self,partial_results):
        self.partial_group=partial_results

    def __call__(self,subset_i,metric_type="acc"):
        value=[partial_i.selected_acc(subset_i,
                                      metric_type=metric_type)  
                    for partial_i in self.partial_group]
        return np.mean(value)

    def series(self,order,metric_type="acc"):
        subsets= utils.selected_subsets(order,full=False)
        return [self(subset_i,metric_type=metric_type) 
                    for subset_i in subsets]

def eval_exp(in_path,
             ord_path,
             clf_type="class_ens"):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        result_path=f"{in_path}/{clf_type}/results"
        result_group=dataset.read_partial_group(result_path)
        return DynamicSubsets(result_group)
    output_dict=helper(in_path)
    print(output_dict)
    ord_dict=utils.read_json(ord_path)
    print(ord_dict)


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
         ord_path="ord/purity.json") 