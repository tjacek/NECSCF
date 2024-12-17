import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import utils
utils.silence_warnings()
import numpy as np
import argparse
import matplotlib.pyplot as plt
import dataset,ens,exp,pred

def all_exp(json_path,
            n_iters=2):
    acc_dict=utils.read_json(json_path)
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
        plt.xlabel("n_clf")
        plt.ylabel("acc")
        plt.legend()
        plt.show()
        plt.clf() 

if __name__ == '__main__':
    all_exp("acc/base.json")
