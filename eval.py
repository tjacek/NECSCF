import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import utils
utils.silence_warnings()
import numpy as np
import argparse,os.path
import matplotlib.pyplot as plt
from scipy import stats
import dataset,ens,exp,pred,utils

def summary(exp_path):
    acc_dict=get_result(exp_path,acc=True)
    for id_i,acc_i in acc_dict.items():
        if(acc_i):
            print(f"{id_i}-{np.mean(acc_i):.4f}")

def get_result(exp_path,
               acc=True):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        partial_path=f"{in_path}/partial"
        if(not os.path.isdir(partial_path)):
            return None
        results=[dataset.read_partial(path_i) 
            for path_i in utils.top_files(partial_path) ]
        if(acc):
            return [result_i.get_metric("acc") for result_i in results]
        return results
    path_dict=helper(exp_path)
    return utils.to_id_dir(path_dict,index=-1)


def acc_plot(exp_path,
             ord_path):
    ord_dict=utils.read_json(ord_path)
    result_dict=get_result(exp_path=exp_path,
                           acc=False)
    acc_dict={}
    for id_i,card_i in ord_dict.items():
        results_i=result_dict[id_i]
        if(results_i):
            order_i=np.argsort(card_i)
            subsets_i=utils.selected_subsets(order_i,full=True)
#            raise Exception(order_i)
            acc_i=[np.mean([result_k.selected_acc(subset_j)
                       for result_k in results_i]) 
                    for subset_j in subsets_i]
            acc_dict[id_i]=acc_i
            print(acc_i)
    make_plot(acc_dict,
              title="title",
              x_label="n_clf",      
              y_label="acc",
              n_iters=2)        

#def acc_plot(json_path:str,
#             title="acc_plot",
#             n_iters=2):
#    acc_dict=utils.read_json(json_path)
#    make_plot(acc_dict,
#              title=title,
#              x_label="n_clf",      
#              y_label="acc",
#              n_iters=n_iters)

def diff_plot(first_json:str,
              second_json:str,
              title="acc_plot",
              n_iters=2):
    first_dict=utils.read_json(first_json)
    second_dict=utils.read_json(second_json)
    diff_dict={ key_i:[ first_j-second_j  
                    for first_j,second_j in zip(diff_i,second_dict[key_i])]
                        for key_i,diff_i in first_dict.items()}
    make_plot(diff_dict,
              title=title,
              x_label="n_clf",
              y_label="diff",
              n_iters=n_iters)

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
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if(args.summary):
        summary(exp_path=args.exp_path)
    acc_plot(exp_path=args.exp_path,
             ord_path=args.ord_path)
#    acc_plot("acc/reversed_full.json",
#             title="reversed_full")
#    diff_plot("acc/base_full.json","acc/reversed_full.json",
#              title="Low purity - high purity  (full)")
#    stat_test("results/RF","results/class_ens")
