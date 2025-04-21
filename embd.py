import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tqdm import tqdm
import re
from sklearn.neighbors import BallTree
import base,dataset,deep,ens,utils

class NECSCF(object):
    def __init__(self,model,clf_type="RF"):
        self.model=model	
        self.clf_type=clf_type
        self.extractor=None
        self.clfs=[]
    
    def fit(self,X,y):
        full_feats=self.get_embd(X)
        for feat_i in full_feats:
            clf_i=base.get_clf(self.clf_type)
            clf_i.fit(feat_i,y)
            self.clfs.append(clf_i)
        return self

    def partial_predict(self,X):
        full_feats=self.get_embd(X)
        votes=[ clf_i.predict_proba(full_feats[i])
                for i,clf_i in enumerate(self.clfs)]
        return np.array(votes) 
    
    def predict(self,X):
        votes=self.partial_predict(X)
        votes=np.sum(votes,axis=0)
        return np.argmax(votes,axis=1)

    def eval(self,data,split_i):
        train_data_i=data.selection(split_i.train_index)
        self.fit(train_data_i.X,train_data_i.y)
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.partial_predict(test_data_i.X)
        result=dataset.PartialResults(y_true=test_data_i.y,
                                      y_partial=raw_partial_i)
        return result

    def get_embd(self,X):
        raise NotImplementedError()

class MultiNECSCF(NECSCF):
    def get_embd(self,X):
        if(self.extractor is None):
            self.extractor=Model(inputs=self.model.inputs, 
                                 outputs=list(self.get_outputs()))
        cs_feats= self.extractor([X], training=False)
        cs_feats=[ cs_i.numpy() for cs_i in cs_feats]
        full_feats=[np.concatenate([X,cs_i],axis=1) 
                        for cs_i in cs_feats]        
        return full_feats

    def get_outputs(self):
        pattern=re.compile(r"(\D)+_(\d)+_1")
        for layer_i in self.model.layers:
            name_i=layer_i.name
            if(pattern.match(name_i)):
                yield layer_i.output

class SeparNECSCF(NECSCF):
    def get_embd(self,X):
        if(self.extractor is None):
            self.make_extractor()
        cs_feats=[extractor_i([X],training=False).numpy()
                    for extractor_i in self.extractor]
        full_feats=[np.concatenate([X,cs_i],axis=1) 
                        for cs_i in cs_feats] 
        return full_feats

    def make_extractor(self):
        self.extractor=[]
        pattern=re.compile(r"(\D)+_(\d)+_1")
        for model_i in self.model:
            for layer_j in model_i.layers:
                name_j=layer_j.name
                if(pattern.match(name_j)):
                    extractor_i=Model(inputs=model_i.inputs, 
                                 outputs=[layer_j.output])
                    self.extractor.append(extractor_i)
                    continue
        return self.extractor

class EmbdReader(object):
    def __init__(self,ens_type):
        self.ens_type=ens_type
    
    def get_info(self):
        return {"ens":f"NECSCF({self.ens_type})","base_ens":self.ens_type}

class MultiReader(EmbdReader):
    def __call__(self,path):
        model=tf.keras.models.load_model(path,
                                        custom_objects={"loss":deep.WeightedLoss()})
        return MultiNECSCF(model)

class SeparReader(EmbdReader):
    def __call__(self,path):
        models=[]
        for path_i in utils.top_files(path):
            model_i=tf.keras.models.load_model(path_i,
                                               custom_objects={"loss":deep.WeightedLoss()})
            models.append(model_i)
        return SeparNECSCF(models)

class ModelIterator(object):
    def __init__(self,path_dict,reader=None):
        if(reader is None):
            reader=get_reader(path_dict['info.js'])
        self.path_dict=path_dict
        self.reader=reader 

    def __call__(self,start=0,step=10):
        for index in range(step):
            i=start+index
            model_path_i=f"{self.path_dict['models']}/{i}.keras"
            ens_i=self.reader(model_path_i)
            raw_split=np.load(f"{self.path_dict['splits']}/{i}.npz")
            split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
            yield i,ens_i,split_i

    def make_hist(self,data,
                      start=0,
                      step=10,
                      k=10):
        n_cats=data.n_cats()
        hist=PurityHistogram(k,n_cats,n_cats+1)
        for i,model_i,split_i in self(start=start,step=step):
            for j,X_j in enumerate(model_i.get_embd(data.X)):
                data_j=dataset.Dataset(X_j,data.y)
                hist.add_embd(j,data_j,split_i)
        return hist

class PurityHistogram(object):
    def __init__(self,k,n_cats,n_clf):
        self.k=k
        self.arr=[ np.zeros((n_cats,n_cats)) for _ in range(n_clf)]

    def add_embd(self,k,data_k,split):
        hist_k=self.arr[k]
        train_data=data_k.selection(split.train_index)
        test_data=data_k.selection(split.test_index)    
        tree=BallTree(train_data.X)
        indces= tree.query(test_data.X,
                           k=self.k+1,
                           return_distance=False)
        for i,ind_i in enumerate(indces):
            y_i=int(test_data.y[i])
            for j in ind_i:
                y_j=int(train_data.y[j])
                hist_k[y_i][y_j]+=1
        cat_size=np.sum(hist_k,axis=1)
        for j,size_j in enumerate(cat_size):
            hist_k[j]/=size_j

    def print(self):
        for arr_i in self.arr:
            print(arr_i)

    def to_dict(self):
        return {i:np.round(arr_i,4).tolist() 
                for i,arr_i in enumerate(self.arr)}

def get_reader(in_path):
    info_dict=utils.read_json(in_path)
    ens_type=info_dict['ens']
    if(ens.is_separ(ens_type)):
        return SeparReader(ens_type)
    else:
        return MultiReader(ens_type)

def embd_exp(data_path,
             model_path,
             ens_type="class_ens",
             n_iters=100):
    @utils.DirFun({"in_path":0,"out_path":1})
    def helper(in_path,out_path):
        print(in_path)
        single_exp(in_path,
                   out_path,
                   ens_type=ens_type)
    output_dict=helper(data_path,model_path)

def single_exp(data_path,
               out_path,
               ens_type="class_ens"):
    data=dataset.read_csv(data_path)
    path_dict=base.get_paths(out_path=out_path,
                        ens_type=ens_type,
                        dirs=['models','info.js'])
    model_iter=ModelIterator(path_dict)
    embd_dict=base.get_paths(out_path=out_path,
                              ens_type=info_dict['ens'],
                              dirs=['results','info.js'])
    utils.make_dir(embd_dict['ens'])
    utils.make_dir(embd_dict['results'])
    for i,model_i,split_i in tqdm(model_iter(start=0,step=100)):
        result_i=model_i.eval(data,split_i)
        result_i.save(f'{embd_dict['results']}/{i}.npz')
    info_dict=model_iter.reader.get_info()
    utils.save_json(info_dict,embd_dict['info.js'])

def knn_purity_hist(data_path,
                    exp_path,
                    ens_type,
                    n_splits=10,
                    n_iters=10,
                    k=10):
    data=dataset.read_csv(data_path)
    path_dict=base.get_paths(out_path=exp_path,
                             ens_type=ens_type,
                             dirs=['models','info.js'])
    model_iter=ModelIterator(path_dict)
    for i in tqdm(range(n_iters)):
        start_i= i*n_splits
        hist_i=model_iter.make_hist(data,
                                    start=start_i,
                                    step=n_splits,
                                    k=k)
        yield hist_i

def multi_hist(data_path,exp_path,out_path):
    def selector(ens_type):
         return ("ens" in ens_type) and (not "NECSCF" in ens_type)

    @utils.EnsembleFun(out_path="out_path",
                       selector=selector)
    def helper(in_path,out_path):
        raw=in_path.split("/")
        ens_type, data_id= raw[-1],raw[-2]
        hist_iter=knn_purity_hist(data_path=f"{data_path}/{data_id}",
                                  exp_path="/".join(raw[:-1]),
                                  ens_type=ens_type,
                                  n_splits=10,
                                  n_iters=10,
                                  k=10)
        utils.make_dir(out_path)
        for i,hist_i in tqdm(enumerate(hist_iter)):
            out_i=f"{out_path}/{i}"
            utils.save_json(hist_i.to_dict(),out_i)
    utils.make_dir(out_path)
    helper(exp_path,out_path)

def single_hist(data_path,exp_path,ens_type):
    for hist_i in knn_purity_hist(data_path,exp_path,ens_type):
        hist_i.print()

if __name__ == '__main__':
    data='vehicle'
#    single_hist(f"../uci/{data}",f"new_exp/{data}","separ_purity_ens")
    multi_hist("../uci","new_exp","new_eval/purity")
#embd_exp("../uci","new_exp")
#simple_exp(f"../uci/{data}",f"new_exp/{data}")