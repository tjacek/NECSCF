import numpy as np
import argparse
import dataset,pred,utils,ens

def all_exp(data_path,exp_path,json_path):
    purity_dict=utils.read_json(json_path)
#    @utils.dir_fun
    def helper(data_path,exp_path):
        name=data_path.split("/")[-1]
        order= purity_dict[name]
        clf_selection=selection(order)
        data=dataset.read_csv(data_path)
        ens_factory=ens.ClassEnsFactory()
        ens_factory.init(data)
        acc=[]
        for split_i,clf_i in pred.model_iter(exp_path,ens_factory):
            result_i=split_i.pred(data,clf_i)
            acc.append(result_i.get_acc())
            print(acc[-1])
    helper(data_path,exp_path)   

def selection(order):
    return [order[:i+1] for i in range(len(order))]

if __name__ == '__main__':
    all_exp("../uci/wine-quality-red",
            'test_exp/wine-quality-red',
            "purity.json")
