import numpy as np
import os.path
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
import base,ens_depen,dataset,utils

class NNDesc(object):
    def __init__(self,ids,
                      desc,
                      targets,
                      name_dict):
        self.ids=ids
        self.desc=desc
        self.targets=targets
        self.name_dict=name_dict

    def get_pairs(self,target_id:str):
        y=self.targets[target_id]
        n_cats=max(y)+1
        pairs=[[] for _ in range(n_cats)]
        for i,y_i in enumerate(y):
            pairs_i=(self.ids[i], self.desc[i])
            pairs[y_i].append(pairs_i)
        return pairs

    def transform(self,alg):
        alg.fit(self.desc)
        self.desc=alg.transform(self.desc)

    def as_data(self,target_id="ens",
                select=None):
        X,y=self.desc,self.targets[target_id]
        if(not (select is None)):
            index=(self.targets["ens"]==self.ens_id(select))
            X,y=X[index],y[index]
        return dataset.Dataset(X=X,y=y)

    def ens_id(self,ens_type):
        for i,ens_i in enumerate(self.name_dict):
            if(ens_i==ens_type):
                return i

def read_nn(in_path):
    name_dict,desc,ids=[],[],[]
    targets={key_i:[] for key_i in ["ens","cat","iter"]}
    for i,path_i in enumerate(utils.top_files(in_path)):
        ens_i=path_i.split("/")[-1]
        name_dict.append(ens_i)
        for j,path_j in enumerate(utils.top_files(path_i)):
            dict_j=utils.read_json(path_j)
            for cat_k,hist_k in dict_j.items():
                id_k=f"{i}_{j}_{cat_k}"
                ids.append(id_k)
                vector_k=np.array(hist_k).flatten()
                desc.append(vector_k)
                targets['ens'].append(i)
                targets['iter'].append(j)
                targets['cat'].append(int(cat_k))
    y_separ,y_weights=[],[]
    for i in targets['ens']:
        y_separ.append(int('separ' in  name_dict[i]))
        y_weights.append(int('purity' in  name_dict[i]))
    targets['separ']=y_separ
    targets['weights']=y_weights
    targets={key_i:np.array(target_i) 
        for key_i,target_i in targets.items()}
    return NNDesc(ids,np.array(desc),targets,name_dict)

def nn_desc_eval(in_path,
                 target_id="weights",
                 clf_type="RF",
                 select=None,
                 binary=False,
                 latex=False):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        nn_desc_i=read_nn(in_path)
        data_i=nn_desc_i.as_data(target_id,select)
        protocol=base.UnaggrSplit(n_splits=3,n_repeats=1)
        data_split=base.DataSplits(data= data_i,
                                   splits=list(protocol.get_split(data_i)))
        if(binary):
            n_cats=data_i.n_cats()
            balance=[]
            for j in range(n_cats):
                y_j=np.zeros(data_i.y.shape)
                y_j[data_i.y==j]=1
                data_j=dataset.Dataset(X=data_i.X,
                                       y=y_j)
                data_split.data=data_j
                results=data_split.basic_eval(clf_type)
                balance.append(np.mean(results.get_metric("f1-binary")))
            return balance
        else:
            results=data_split.basic_eval(clf_type)
            random=(100/data_i.n_cats())
            return np.mean(results.get_metric("acc")),random
    output=helper(in_path)
    if(binary):
        def helper(pair_i):
            key_i,value_i=pair_i
            id_i=key_i.split('/')[-1]
            return [id_i]+value_i
        cols,offset=["id"],"-"
    else:
        def helper(pair_i):
            key_i,(acc_i,base_i)=pair_i
            id_i=key_i.split('/')[-1]
            return [id_i,np.round(base_i,2),100*np.round(acc_i,4)]
        cols,offset=["id","base","acc"],None
    df=dataset.make_df(helper=helper,
                       iterable=output.items(),
                       cols=cols,
                       offset=offset)
    if(latex):
        print(df.to_latex())
    else:
        df.print()

@utils.DirFun({'in_path':0,'out_path':1})
def nn_desc_plot(in_path,
                 out_path=None,
                 transform_type="lle",
                 target_id='iter'):
    nn_desc=read_nn(in_path)
    if(target_id=="outliners"):
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(nn_desc.desc)
        y = clf.predict(nn_desc.desc)
        y[y==(-1)]=0
        nn_desc.targets[target_id]=y
    reduction=get_reduction(transform_type)
    nn_desc.transform(reduction)
    series=nn_desc.get_pairs(target_id)
    txt_plot(series,
             title=transform_type,
             out_path=out_path)

def get_reduction(type):
    if(type=='lle'):
        return manifold.LocallyLinearEmbedding(n_neighbors=5,
                                          n_components=2,
                                          method='standard')
    return PCA(n_components=2)

def txt_plot(series,
             labels=None,
             title="",
             out_path=None):
    if(labels is None):
        labels=make_color_map(series)
    plt.figure()
    plt.title(title)
    all_points=[]
    for i, series_i in enumerate(series):
        for id_j,point_j in series_i:
            plt.text(point_j[0], 
                     point_j[1], 
                     id_j,
                     color=labels(i),
                     fontdict={'weight': 'bold', 'size': 9})
            all_points.append(point_j)
    all_points=np.array(all_points)
    min_point=np.amin(all_points,axis=0)
    max_point=np.amax(all_points,axis=0)
    plt.xlim((min_point[0],max_point[0]))
    plt.ylim((min_point[1],max_point[1]))
    if(out_path):
        plt.savefig(out_path)
    else:
        plt.show()

def make_color_map(series):
    n_cats=len(series)
    if(n_cats==2):
        binary_colors=['r','b']
        return lambda i:binary_colors[i]
    cat2col= np.arange(n_cats)
    np.random.shuffle(cat2col)
    def color_helper(i):
        return plt.cm.tab20(cat2col[int(i)])
    return color_helper

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
#    nn_desc_eval("new_eval/purity")
    knn_purity("../uci",k=10)
#    history_epoch("new_exp",
#                  ens_type="class_ens",
#                  out_path="new_eval/n_epochs/class_ens")
#    history_acc("new_exp","ord/total_acc.json")