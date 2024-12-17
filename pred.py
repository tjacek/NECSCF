import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gc,json
import multiprocessing
import base,dataset,ens,exp

def order_exp(data_path:str,
              exp_path:str,
              json_path:str,
              out_path:str):
    utils.make_dir(out_path)
    exp_types={"base_full":(True,False),
               "base":(False,False),
               "reversed":(False,True),
               "reversed_full":(True,True)}
    for name_i,(full_i,reverse_i) in exp_types.items():
        order_pred(data_path=data_path,
                   exp_path=exp_path,
                   json_path=json_path,
                   out_path=f"{out_path}/{name_i}.json",
                   full=full_i,
                   reverse=reverse_i)

def order_pred(data_path:str,
               exp_path:str,
               json_path:str,
               out_path:str,
               full=False,
               reverse=False):
    card_dict=utils.read_json(json_path)
    def helper(data_path,exp_path,queue):
        name=data_path.split("/")[-1]
        print(name)
        card= card_dict[name]
        order=np.argsort(card)
        if(reverse):
            order=np.flip(order)
        clf_selection=utils.selected_subsets(order.tolist(),
                                             full=full)
        data=dataset.read_csv(data_path)
        ens_factory=ens.ClassEnsFactory()
        ens_factory.init(data)
        acc=[[] for _ in clf_selection]
        m_iter=  model_iter(exp_path,ens_factory)
        for i in tqdm(range(10)):
            split_i,clf_i=next(m_iter)
#        for split_i,clf_i in tqdm(model_iter(exp_path,ens_factory)):
            for j,subset_j in enumerate(clf_selection):
                clf_j=exp.SelectedEns(clf_i,subset_j)
                acc[j].append(split_i.pred(data,clf_j).get_acc())
        acc=np.array(acc)
        queue.put(np.mean(acc,axis=1).tolist())
    acc_dict={}#helper(data_path,exp_path)
    for path_i in utils.top_files(data_path):
        id_i=path_i.split("/")[-1]
        exp_i=f"{exp_path}/{id_i}"
        queue = multiprocessing.Queue()
        p_i=multiprocessing.Process(target=helper, 
                                    args=(path_i,exp_i,queue))
        p_i.start()
        p_i.join()
        acc_dict[id_i]=queue.get()
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(acc_dict, f, ensure_ascii=False, indent=4)
    return acc_dict

#def selection_pred(data_path,model_path):
#    data=dataset.read_csv(data_path)
#    ens_factory=ens.ClassEnsFactory()
#    ens_factory.init(data)
#    subsets=list(exp.iter_subsets(data.n_cats()+1))
#    results=[[] for _ in subsets]
#    for split_i,clf_i in tqdm(model_iter(model_path,ens_factory)):
#        for j,subset_j in enumerate(subsets):
#            clf_j=exp.SelectedEns(clf_i,subset_j)
#            results[j].append(split_i.pred(data,clf_j))
#    for j,subset_j in enumerate(subsets):   
#        acc_j=np.mean([ result.get_acc() for result in results[j]])
#        print(f"{subset_j}:{acc_j}")

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

if __name__ == '__main__':
    acc_dir=order_exp(data_path="../uci",
                      exp_path="exp_deep",
                      json_path="ord/purity.json",
                      out_path="acc")
    print(acc_dir)