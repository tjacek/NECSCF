import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import base,dataset,ens,exp

def selection_pred(data_path,model_path):
    data=dataset.read_csv(data_path)
    ens_factory=ens.ClassEnsFactory()
    ens_factory.init(data)
    subsets=list(exp.iter_subsets(data.n_cats()+1))
    results=[[] for _ in subsets]
    for split_i,clf_i in tqdm(model_iter(model_path,ens_factory)):
        for j,subset_j in enumerate(subsets):
            clf_j=exp.SelectedEns(clf_i,subset_j)
            results[j].append(split_i.pred(data,clf_j))
    for j,subset_j in enumerate(subsets):   
        acc_j=np.mean([ result.get_acc() for result in results[j]])
        print(f"{subset_j}:{acc_j}")

def simple_pred(data_path,exp_path):
    data=dataset.read_csv(data_path)
    ens_factory=ens.ClassEnsFactory()
    ens_factory.init(data)
    acc=[]
    for split_i,clf_i in tqdm(model_iter(exp_path,ens_factory)):
        result_i=split_i.pred(data,clf_i)
        acc.append(result_i.get_acc())
    return np.mean(acc)

def model_iter(exp_path,ens_factory):
    split_path=f"{exp_path}/splits"
    model_path=f"{exp_path}/class_ens"
    for i,path_i in enumerate(utils.top_files(model_path)):
        split_i=f"{split_path}/{i}.npz"
        raw_split=np.load(split_i)
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        model_i=tf.keras.models.load_model(f"{model_path}/{i}.keras")
        clf_i=ens_factory()
        clf_i.model=model_i
        yield split_i,clf_i

#@utils.DirFun({"data_path":0,"model_path":1},
#              input_arg='data_path')
#def simple_pred(data_path,model_path):
#    print(data_path)
#    data=dataset.read_csv(data_path)
#    ens_factory=ens.ClassEnsFactory()
#    ens_factory.init(data)
#    acc=[]
#    for split_i,clf_i in model_iter(model_path,ens_factory):
#        result_i=split_i.pred(data,clf_i)
#        acc.append(result_i.get_acc())
#    return np.mean(acc)

#def model_iter(model_path,ens_factory):
#    for path_i in utils.top_files(model_path):
#        raw_split=np.load(f"{path_i}/split.npz")
#        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
#                                       test_index=raw_split["arr_1"])
#        model_i=tf.keras.models.load_model(f"{path_i}/model.h5")
#        clf_i=ens_factory()
#        clf_i.model=model_i
#        yield split_i,clf_i

if __name__ == '__main__':
    acc_dir=simple_pred(data_path="../uci/cleveland",
                        exp_path="exp_deep/cleveland")
    print(acc_dir)