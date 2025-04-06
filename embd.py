import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
import re
import base,dataset,deep

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
    
    def predict(self,X):
        full_feats=self.get_embd(X)
        votes=[ clf_i.predict_proba(full_feats[i])
                for i,clf_i in enumerate(self.clfs)]
        votes=np.array(votes)
        votes=np.sum(votes,axis=0)
        return np.argmax(votes,axis=1)

    def eval(self,data,split_i):
        result,_=split_i.eval(data,self)
        return result

    def get_embd(self,X):
        raise NotImplementedError()

class MultiNECSCF(NECSCF):
    def get_embd(self,X):
        if(self.extractor is None):
            self.extractor=Model(inputs=self.model.inputs, 
                                 outputs=list(self.get_outputs()))
        cs_feats= self.extractor(X, training=False)
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

class MultiReader(object):
    def __init__(self,ens_type):
        self.ens_type=ens_type

    def __call__(self,path):
        model=tf.keras.models.load_model(path,
                                        custom_objects={"loss":deep.WeightedLoss()})
        return MultiNECSCF(model)

def embd_exp(data_path,model_path):
    data=dataset.read_csv(data_path)
    acc=[]
    for i,model_i,split_i in read_models(model_path):
        result_i=model_i.eval(data,split_i)	
        acc.append(result_i.get_acc())
    print(np.mean(acc))

def read_models(in_path,
                ens_type="class_ens",
                start=0,
                step=10):
    split_path=f"{in_path}/splits"
    model_path=f"{in_path}/{ens_type}/models"
    reader=MultiReader(ens_type)
    for index in range(step):
        i=start+index
        model_path_i=f"{model_path}/{i}.keras"
        print(model_path_i)
#        model_i=tf.keras.models.load_model(model_path_i,
#                                           custom_objects={"loss":deep.WeightedLoss()})
        ens_i=reader(model_path_i)
        raw_split=np.load(f"{split_path}/{i}.npz")
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        yield i,ens_i,split_i

data='wine-quality-red'
embd_exp(f"../uci/{data}",f"new_exp/{data}")