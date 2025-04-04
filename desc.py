import numpy as np
import os.path
from sklearn.neighbors import BallTree
import dataset,utils

def purity_hist(in_path,k=10):
    data_i=dataset.read_csv(in_path)
    tree=BallTree(data_i.X)
    indces= tree.query(data_i.X,
                           k=k+1,
                          return_distance=False)
    n_cats=data_i.n_cats()
    hist=np.zeros((n_cats,n_cats))
    sizes=np.zeros(n_cats)
    for i,ind_i in enumerate(indces):
        point_i=int(data_i.y[i])
        sizes[point_i]+=1
        for ind_j in ind_i[1:]:
            near_j=int(data_i.y[ind_j])
            hist[point_i][near_j]+=1
    hist/=k
    for i,size_i in enumerate(sizes):
        hist[i,:]/=size_i
    return hist

def history_epoch(exp_path,
                  ens_type="purity_ens"):
    @utils.EnsembleFun(selector=ens_type)
    def helper(in_path):
        history_path=f"{in_path}/history"
        all_history=[ utils.read_json(path_i)
                for path_i in utils.top_files(history_path)]
        if(not "separ" in ens_type):
            all_history=[ [history_i] for history_i in all_history]
        desc_vector,n_clfs=[],len(all_history[0])
        for i in range(n_clfs):
            epochs=[history_j[i]['n_epochs'] for history_j in all_history]
            desc_vector.append(np.mean(epochs))
        return desc_vector
    output= helper(exp_path)
    for data_i,_,vector_i in output:
        print(data_i)
        print(vector_i)



def knn_purity(in_path,k=10):
    helper=utils.DirFun({"in_path":0})(purity_hist)
    output_dict=helper(in_path,k)
    for name_i,value_i in output_dict.items():
        value_i=np.round(value_i,4)
        print(name_i)
        print(value_i)

def history_acc(exp_path,out_path=None):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        for path_i in utils.top_files(in_path):
            history_i=f"{path_i}/history"
            if(os.path.exists(history_i)):
                hist=[utils.read_json(path_j) 
                            for path_j in utils.top_files(history_i)]
                keys=[key_j for key_j in hist[0].keys()
                        if(not "loss" in key_j)]
                hist_dict={ key_j:[hist_k[key_j] for hist_k in hist]
                             for key_j in keys}
                hist_dict={ key_j: (round(np.mean(value_j),4),
                                    round(np.std(value_j),4))
                             for key_j,value_j in hist_dict.items()}
                return hist_dict
    output_dict=helper(exp_path)
    for name_i,dict_i in output_dict.items():
        print(name_i)
        print(dict_i)
    if(out_path):
        ord_dict={}
        for name_i,dict_i in output_dict.items():
            keys={ utils.extract_number(key_j):key_j
                    for key_j in dict_i
                        if("acc" in key_j)}
            values=[ dict_i[keys[k]][0]
                        for k in range(len(keys))]
            values=z_score(values)
            ord_dict[name_i.split("/")[-1]]=values
        utils.save_json(ord_dict,out_path)

def z_score(values):
    values=np.array(values)
    values-=np.mean(values)
    values/=np.std(values)
    values=np.round(values,4)
    return values.tolist()

if __name__ == '__main__':
    history_epoch("new_exp")
#    history_acc("new_exp","ord/total_acc.json")