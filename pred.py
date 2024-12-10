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

def simple_pred(data_path,model_path):
    data=dataset.read_csv(data_path)
    ens_factory=ens.ClassEnsFactory()
    ens_factory.init(data)
    acc=[]
    for split_i,clf_i in model_iter(model_path,ens_factory):
        result_i=split_i.pred(data,clf_i)
        acc.append(result_i.get_acc())
    print(np.mean(acc))

def model_iter(model_path,ens_factory):
    for path_i in utils.top_files(model_path):
        raw_split=np.load(f"{path_i}/split.npz")
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        model_i=tf.keras.models.load_model(f"{path_i}/model.h5")
        clf_i=ens_factory()
        clf_i.model=model_i
        yield split_i,clf_i

#def all_pred(in_path,out_path):
#    metric=utils.get_metric('acc')
#    @utils.dir_fun
#    def helper(in_path,out_path):
#        exp_i=base.read_exp(in_path)
#        y_true,y_pred=nescf_eval(exp_i)
#        print(f'{out_path} Acc:{metric(y_true,y_pred):.4f}')
#        np.savez(file=out_path,
#                 true=y_true,
#                 pred=y_pred)
#    helper(in_path,out_path)

#def nescf_eval(exp):
#    extractor=exp.make_extractor() 
#    train,test=exp.split.extract(extractor=extractor,
#                                 use_valid=False)
#    (x_train,y_train),(x_test,y_test)=train,test
#    y_pred=base.simple_necscf(x_train=x_train,
#                              y_train=y_train,
#                              x_test=x_test,
#                              clf_type="RF")
#    return y_test,y_pred

if __name__ == '__main__':
    selection_pred(data_path="../uci/wine-quality-red",
                   model_path="exp")