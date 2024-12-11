import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import utils
utils.silence_warnings()
import numpy as np
import argparse
import matplotlib.pyplot as plt
import dataset,ens,exp,pred

def all_exp(data_path,
            exp_path,
            json_path,
            out_path,
            n_iters=2):
    purity_dict=utils.read_json(json_path)
    @utils.DirFun({"data_path":0,"model_path":1},
                  input_arg='data_path')
    def helper(data_path,exp_path):
        name=data_path.split("/")[-1]
        print(name)
        order= purity_dict[name]
        clf_selection=selection(order)
        data=dataset.read_csv(data_path)
        ens_factory=ens.ClassEnsFactory()
        ens_factory.init(data)
        acc=[[] for _ in order]
        for split_i,clf_i in pred.model_iter(exp_path,ens_factory):
            for j,subset_j in enumerate(clf_selection):
                clf_j=exp.SelectedEns(clf_i,subset_j)
                acc[j].append(split_i.pred(data,clf_j).get_acc())
        acc=np.array(acc)
        return np.mean(acc,axis=1)
    acc=helper(data_path,exp_path)   
    print(acc)
    utils.make_dir(out_path)
    all_subplots=[[] for k in range(n_iters)]
    for i,name_i in enumerate(acc.keys()):
        all_subplots[(i%n_iters)].append(name_i)
    for i,subplot_i in enumerate(all_subplots):
        _, ax_k = plt.subplots()
        for name_j in subplot_i:
            acc_j=acc[name_j]
            x_order=np.arange(acc_j.shape[0])+1
            ax_k.plot(x_order,acc_j,
                      label=name_j.split("/")[-1])
        plt.xlabel("n_clf")
        plt.ylabel("acc")
        plt.legend()
        plt.savefig(f"{out_path}/{i}")
        plt.show()
        plt.clf() 
    
#    for k in range(n_iters):
#        _, ax_k = plt.subplots()
#        all_subplots.append(ax_k)
#    for i,(name_i,acc_i) in enumerate(acc.items()):
#        x_order=np.arange(acc_i.shape[0])+1
#        k=(i % n_iters)
#        all_subplots[k].plot(x_order,acc_i,
#                             label=name_i.split("/")[-1])
#    for k,ax_k in enumerate(all_subplots):
#        plt.xlabel("n_clf")
#        plt.ylabel("acc")
#        ax_k.legend()
#    plt.savefig(f"{out_path}/{k}")
#    plt.show()

def selection(order):
    return [order[:i+1] for i in range(len(order))]

if __name__ == '__main__':
    all_exp("../uci/",#wine-quality-red,
            "full_models",#wine-quality-red',
            "purity.json",
            "acc_plot")
