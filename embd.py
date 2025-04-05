import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
import re
import base,dataset,deep

class NECSCF(object):
    def __init__(self,model,clf_type="RF"):
        self.model=model	
        self.clf_type=clf_type
        self.clfs=[]

    def get_outputs(self):
        pattern=re.compile(r"(\D)+_(\d)+_1")
        for layer_i in self.model.layers:
            name_i=layer_i.name
            if(pattern.match(name_i)):
                yield layer_i.output
    
    def fit(self,X,y):
        extractor=Model(inputs=self.model.inputs, 
                        outputs=list(self.get_outputs()))
#        extractor.summary()
        cs_feats= extractor(X, training=False)
        raise Exception([cs_i.shape for cs_i in cs_feats])

    def eval(self,data,split_i):
    	clf=split_i.eval(data,self)

def embd_exp(data_path,model_path):
    data=dataset.read_csv(data_path)
    for i,model_i,split_i in read_models(model_path):
        model_i.eval(data,split_i)	

def read_models(in_path,
                ens_type="class_ens",
                start=0,
                step=10):
    split_path=f"{in_path}/splits"
    model_path=f"{in_path}/{ens_type}/models"
    for index in range(step):
        i=start+index
        model_path_i=f"{model_path}/{i}.keras"
        print(model_path_i)
        model_i=tf.keras.models.load_model(model_path_i,
                                           custom_objects={"loss":deep.WeightedLoss()})
        raw_split=np.load(f"{split_path}/{i}.npz")
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        yield i,NECSCF(model_i),split_i

embd_exp("../uci/cmc","new_exp/cmc")