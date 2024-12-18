import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import utils
utils.silence_warnings()
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import dataset,ens,exp,pred

#class Cond(object):
#    def __init__(self,thres):
#        self.thres=thres

#    def __call__(self,feats):
#        feats=np.array(feats)
#        return sum((feats<thres)astype(int))>0

def acc_plot(json_path:str,
            n_iters=2):
    acc_dict=utils.read_json(json_path)
    make_plot(acc_dict,
              x_label="n_clf",      
              y_label="acc",
              n_iters=n_iters)

def diff_plot(first_json:str,
              second_json:str,
              n_iters=2):
    first_dict=utils.read_json(first_json)
    second_dict=utils.read_json(second_json)
    diff_dict={ key_i:[ first_j-second_j  
                    for first_j,second_j in zip(diff_i,second_dict[key_i])]
                        for key_i,diff_i in first_dict.items()}
    make_plot(diff_dict,
              x_label="n_clf",
              y_label="diff",
              n_iters=n_iters)

def make_plot(acc_dict,
              x_label="n_clf",
              y_label="acc",
              n_iters=2):
    all_subplots=[[] for k in range(n_iters)]
    for i,name_i in enumerate(acc_dict.keys()):
        all_subplots[(i%n_iters)].append(name_i)
    for i,subplot_i in enumerate(all_subplots):
        _, ax_k = plt.subplots()
        for name_j in subplot_i:
            acc_j=acc_dict[name_j]
            x_order=np.arange(len(acc_j))+1
            ax_k.plot(x_order,acc_j,
                      label=name_j.split("/")[-1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        plt.clf() 

def stat_test(x_path,y_path):
    @utils.DirFun({"x_path":0,"y_path":1},input_arg='x_path')
    def helper(x_path,y_path):
        print((x_path,y_path))
        x_results=dataset.read_result_group(x_path)
        y_results=dataset.read_result_group(y_path)
        diff=np.mean(x_results.get_acc())-np.mean(y_results.get_acc())
        pvalue=stats.ttest_ind(x_results.get_acc(),y_results.get_acc(),
                               equal_var=False)[1]
        return diff,pvalue
    stat_dict=helper(x_path,y_path)
    print(stat_dict)

if __name__ == '__main__':
#    diff_plot("acc/base.json","acc/reversed.json")
    stat_test("results/RF","results/class_ens")