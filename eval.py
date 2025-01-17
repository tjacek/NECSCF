import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import utils
utils.silence_warnings()
import numpy as np
import argparse,os.path
import matplotlib.pyplot as plt
from scipy import stats
import dataset,ens,exp,pred,utils

class SubsetEval(object):
    def __init__(self,ord_dict,result_dict):
        self.ord_dict=ord_dict
        self.result_dict=result_dict

    def iter(self,fun,order=True):
        fun_dict={}
        for key_i in self.keys():
            order_i=self.ord_dict[key_i]
            partial_i=self.result_dict[key_i]
            if(order):
                order_i=np.argsort(order_i)
            fun_dict[key_i]=fun(partial_i,order_i)
        return fun_dict

    def keys(self):
        return self.result_dict.keys()

def read_subset_eval(ord_path,exp_path):
    ord_dict=utils.read_json(ord_path)
    result_dict=pred.get_result(exp_path=exp_path,
                                acc=False)
    return SubsetEval(ord_dict=ord_dict,
                      result_dict=result_dict)

def summary(exp_path):
    acc_dict=pred.get_result(exp_path,acc=True)
    for id_i,acc_i in acc_dict.items():
        if(acc_i):
            print(f"{id_i}-{np.mean(acc_i):.4f}")

def acc_plot(exp_path,
             ord_path,
             reverse=True):
    subset_eval=read_subset_eval(ord_path,exp_path)
    def helper(partial_i,order_i):
        if(reverse):
            order_i=np.flip(order_i)
        return partial_i.order_acc(order_i,full=True)
    acc_dict=subset_eval.iter(helper,order=True)    
    make_plot(acc_dict,
              title="Selection knn-purity",
              x_label="n_clf",      
              y_label="acc",
              n_iters=2)        

def diff_plot(exp_path,
             ord_path,
             reverse=True):
    subset_eval=read_subset_eval(ord_path,exp_path)
    def helper(partial_i,order_i):   
        acc=partial_i.order_acc(order_i,full=True)
        flip_i=np.flip(order_i)
        flip_acc= partial_i.order_acc(flip_i,full=True)
        return acc - flip_acc
    acc_dict=subset_eval.iter(helper,order=True)   
    make_plot(acc_dict,
              title="title",
              x_label="n_clf",      
              y_label="acc",
              n_iters=2)        

def cum_sum(card_i,order_i):
    cum,total=[],0.0
    for j in order_i:
        total+= card_i[j]
        cum.append(total)
    return cum

def purity_plot(exp_path,
                ord_path):
    subset_eval=read_subset_eval(ord_path,exp_path)
    def helper(partial_i,purity_i):
        order_i=np.argsort(purity_i)
        acc_i=partial_i.order_acc(order_i,full=True)
        cum_i=cum_sum(purity_i,order_i)
        return cum_i,acc_i[:-1]
    acc_dict=subset_eval.iter(helper,order=False)   
    scatter_plot(acc_dict)

def single_plot(exp_path,
                ord_path):
    subset_eval=read_subset_eval(ord_path,exp_path)
    def helper(partial_i,purity_i):
        acc_i=partial_i.indv_acc()
        return purity_i,acc_i
    acc_dict=subset_eval.iter(helper,order=False)   
    scatter_plot(acc_dict)

def scatter_plot(acc_dict):
    for key_i,(x,y) in acc_dict.items():
        plt.title(key_i)
        plt.xlabel("purity")
        plt.ylabel("acc")
        plt.scatter(x, y)
        plt.show()
        plt.clf()

def make_plot(acc_dict,
              title="acc_plot",
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
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        plt.clf() 

def get_plot_fun(plot_type:str):
    if(args.type=="diff"):
        return diff_plot
    if(args.type=="purity"):
        return purity_plot
    if(args.type=="single"):
        return single_plot 

def stat_test(x_path,y_path):
    @utils.DirFun({"x_path":0,"y_path":1},input_arg='x_path')
    def helper(x_path,y_path):
        print((x_path,y_path))
        x_acc=dataset.read_result_group(x_path).get_acc()
        y_acc=dataset.read_result_group(y_path).get_acc()
        mean_x,mean_y=np.mean(x_acc),np.mean(y_acc)
        diff= mean_x-mean_y
        pvalue=stats.ttest_ind(x_acc,y_acc,
                               equal_var=False)[1]
        return mean_x,mean_y,diff,pvalue
    stat_dict=helper(x_path,y_path)
    for name_i,stats_i in stat_dict.items():
        print(name_i)
        print(",".join([f"{stat_j:.4f}" for stat_j in stats_i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="exp_deep")
    parser.add_argument("--ord_path", type=str, default="ord/purity.json")
    parser.add_argument('--type', default='single', choices=['acc', 'diff', 'purity','single']) 
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if(args.summary):
        summary(exp_path=args.exp_path)
    plot_fun=get_plot_fun(plot_type=args.type)
    plot_fun(exp_path=args.exp_path,
             ord_path=args.ord_path)
#    stat_test("results/RF","results/class_ens")
