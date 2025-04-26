import numpy as np
import os.path
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
import ens_depen,dataset,utils

class PurityVectors(object):
    def __init__(self,ids,vectors):
        self.ids=ids
        self.vectors=vectors
    
    def sub_mean(self):
        mean=np.mean(self.vectors,axis=0)
        self.vectors-=mean
        return self

    def reduce(self,alg):
        new_feats=alg.transform(self.vectors)
        return [ (id_i,new_feats[i]) 
                    for i,id_i in enumerate(self.ids)]

    def outliners(self,get_id=True):
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(self.vectors)
        y_pred = clf.predict(self.vectors)
        n_outliners=len( y_pred[y_pred==(-1)])
        if(get_id):
            out_ids=[]
            for i,pred_i in enumerate(y_pred):
                if(pred_i==(-1)):
                    out_ids.append(self.ids[i])
            return n_outliners,out_ids
        return n_outliners
        
def detect_outliners(in_path):
    @utils.EnsembleFun(selector=lambda ens_id:True)
    def helper(in_path):
        return read_purity(in_path)
    output=helper(in_path)
    for data_i,ens_i,purity_i in output:
        n_out=purity_i.outliners()
        print((data_i,ens_i,n_out)) 

def read_purity(in_path):
    ids,vectors=[],[]
    for i,path_i in enumerate(utils.top_files(in_path)):
        dict_i=utils.read_json(path_i)
        for cat_j,hist_j in dict_i.items():
            id_ij=f"{i}_{cat_j}"
            vector_j=np.array(hist_j).flatten()
            ids.append(id_ij)
            vectors.append(vector_j)
    return PurityVectors(ids,np.array(vectors))#.sub_mean()

def pca_purity(in_path,
               transform_type="pca"):
    vector_dict={path_i.split("/")[-1]:read_purity(path_i)
            for path_i in utils.top_files(in_path)}
    all_vectors=[]
    for purity_i in vector_dict.values():
        all_vectors+=purity_i.vectors.tolist()
    reduction=get_reduction(transform_type)
    reduction.fit(all_vectors)
    series=[]
    for _,purity_i in vector_dict.items():
        pairs_i= purity_i.reduce(reduction)
        series.append(pairs_i)
    txt_plot(series,
             labels=['r','g','b','y'],
             title=transform_type)

def txt_plot(series,
             labels,
             title):
    plt.figure()
    plt.title(title)
    all_points=[]
    for i, series_i in enumerate(series):
        for id_j,point_j in series_i:
            plt.text(point_j[0], 
                     point_j[1], 
                     id_j,
                     color=labels[i],
                     fontdict={'weight': 'bold', 'size': 9})
            all_points.append(point_j)
    all_points=np.array(all_points)
    min_point=np.amin(all_points,axis=0)
    max_point=np.amax(all_points,axis=0)

#    raise Exception(min_point)
    plt.xlim((min_point[0],max_point[0]))
    plt.ylim((min_point[1],max_point[1]))
    plt.show()

def get_reduction(type):
    if(type=='lle'):
        return manifold.LocallyLinearEmbedding(n_neighbors=5,#n_neighbors, 
                                          n_components=2,#n_components,
                                          method='standard')
    return PCA(n_components=2)
def history_epoch(exp_path,
                  ens_type="separ_class_ens",
                  out_path=None):
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
    output_dict={}
    for data_i,_,vector_i in output:
        if(len(vector_i)>1):
            vector_i=np.mean(vector_i)
        else:
            vector_i=vector_i[0]
        output_dict[data_i]=vector_i
    if(out_path):
        utils.save_json(output_dict,out_path)
    print(output_dict)

def knn_purity(in_path,k=10):
    helper=utils.DirFun({"in_path":0})(ens_depen.purity_hist)
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
    pca_purity("new_eval/purity/satimage")
#    detect_outliners("new_eval/purity")#"vehicle/class_ens")
#    history_epoch("new_exp",
#                  ens_type="class_ens",
#                  out_path="new_eval/n_epochs/class_ens")
#    history_acc("new_exp","ord/total_acc.json")