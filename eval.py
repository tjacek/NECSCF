import numpy as np
import argparse
import dataset,ens,exp,pred,utils

def all_exp(data_path,exp_path,json_path):
    purity_dict=utils.read_json(json_path)
    @utils.DirFun({"data_path":0,"model_path":1},
                  input_arg='data_path')
    def helper(data_path,exp_path):
        name=data_path.split("/")[-1]
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

def selection(order):
    return [order[:i+1] for i in range(len(order))]

if __name__ == '__main__':
    all_exp("../uci/",#wine-quality-red,
            "test_exp",#wine-quality-red',
            "purity.json")
