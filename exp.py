import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import base,dataset,deep,utils


class ClfFactory(object):
    def __init__(self,clf_type="RF"):
        self.clf_type=clf_type
    
    def init(self,data):
        pass

    def __call__(self):
        return get_clf(self.clf_type)

def get_clf(clf_type):
    if(clf_type=="RF"): 
        return RandomForestClassifier(class_weight="balanced")#_subsample")
    if(clf_type=="LR"):
        return LogisticRegression(solver='liblinear')
    raise Exception(f"Unknow clf type:{clf_type}")

class ClassEnsFactory(object):
    def __init__(self,hyper_params=None):
        if(hyper_params is None):
           hyper_params={'layers':1,'units_0':2,'batch':True}
        self.params=None
        self.hyper_params=hyper_params
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'class_weights':dataset.get_class_weights(data.y) }

    def __call__(self):
        return ClassEns(params=self.params,
                        hyper_params=self.hyper_params)

class ClassEns(object):
    def __init__(self, params,
                       hyper_params,
                       model=None):
        self.params=params
        self.hyper_params=hyper_params
        self.model = model

    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        if(self.model is None):
            self.model=deep.ensemble_builder(params=self.params,
	                                         hyper_params=self.hyper_params,
	                                         alpha=0.5)
        y=[tf.one_hot(y,depth=self.params['n_cats'])
                for i in range(data.n_cats())]
        self.model.fit(x=X,
        	           y=y,
        	           callbacks=deep.get_callback())

    def predict(self,X):
    	y=self.model.predict(X)
    	y=np.sum(np.array(y),axis=0)
    	return np.argmax(y,axis=1)

def clf_exp(in_path,
            n_splits=10,
            n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=protocol.get_split(data)
    @utils.elapsed_time
    def helper(clf_factory):
        clf_factory.init(data)
        results=[]
        for split_k in splits:
            clf_k=clf_factory()
            results.append(split_k.eval(data,clf_k))
        acc=np.mean([result_j.get_acc() for result_j in results])
        balance=np.mean([result_j.get_balanced() for result_j in results])
        return acc,balance
    clfs={'RF':ClfFactory('RF'),
           'class_ens':ClassEnsFactory()}
    acc_dict,balance_dict={},{}
    for clf_type_i,clf_i in clfs.items():
        acc_i,balance_i=helper(clf_i)
        acc_dict[clf_type_i]=acc_i
        balance_dict[clf_type_i]=balance_i
    print(acc_dict)
    print(balance_dict)
        

@utils.elapsed_time
def ensemble_exp(in_path,
                 n_splits=10,
                 n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=protocol.get_split(data)
    clf_factory=ClassEnsFactory()
    clf_factory.init(data)
    results=[]
    for split_k in splits:
        clf_k=clf_factory()
        results.append(split_k.eval(data,clf_k))
    acc=np.mean([result_j.get_acc() for result_j in results])
    balance=np.mean([result_j.get_balanced() for result_j in results])
    print(f"{acc},{balance}")

clf_exp(in_path="../uci/wine-quality-red")
#clf.fit(data.X,data.y)
#clf.predict(data.X)
#model.summary()