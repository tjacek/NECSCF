def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import tensorflow as tf
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import base,dataset,deep,utils

class DataSplits(object):
    def __init__(self,data,splits):
        self.data=data
        self.splits=splits
        
    def __call__(self,clf_factory):
        clf_factory.init(self.data)
        results=[]
        for split_k in self.splits:
            clf_k=clf_factory()
            results.append(split_k.eval(self.data,clf_k))
        return dataset.ResultGroup(results)

    def get_clfs(self,clf_factory):
        clf_factory.init(self.data)
        for split_k in self.splits:
            print("OK")
            clf_k=clf_factory()
            yield split_k.fit_clf(self.data,clf_k)

    def pred(self,clfs):
        results=[]
        for split_k,clf_k in zip(self.splits,clfs):
            results.append(split_k.pred(self.data,clf_k))
        return dataset.ResultGroup(results)

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
    def __init__(self,hyper_params=None,selected_classes=None):
        if(hyper_params is None):
           hyper_params={'layers':2, 'units_0':2,
                         'units_1':1,'batch':False,
                         'selected_classes':selected_classes}
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
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.model = model
        self.verbose=verbose

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
        	           callbacks=deep.get_callback(),
                       verbose=self.verbose)

    def predict(self,X):
    	y=self.model.predict(X,
                             verbose=self.verbose)
    	y=np.sum(np.array(y),axis=0)
    	return np.argmax(y,axis=1)

    def select_predict(self,X,select_cats):
        y=self.model.predict(X,
                             verbose=self.verbose)
        y=[y[cat_i] for cat_i in select_cats]
        y=np.sum(np.array(y),axis=0)
        return np.argmax(y,axis=1)       

class SelectedEns(object):
    def __init__(self,ens,select_cats):
        self.ens=ens
        self.select_cats=select_cats

    def predict(self,X):
        return self.ens.select_predict(X=X,
                                       select_cats=self.select_cats)

def clf_exp(in_path,
            n_splits=10,
            n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=DataSplits( data=data,
                       splits=protocol.get_split(data))
     
    selected_classes=selection(data)#[0,1,2]
    clfs={'RF':ClfFactory('RF'),
           'class_ens':ClassEnsFactory(selected_classes=selected_classes)}
    acc_dict,balance_dict={},{}
    for clf_type_i,clf_i in clfs.items():
        results=splits(clf_i)
        acc_dict[clf_type_i]=np.mean(results.get_acc())
        balance_dict[clf_type_i]=np.mean(results.get_balanced() )
    print(acc_dict)
    print(balance_dict)

def selection(data):
    sizes=data.class_percent()
    return [ i for i,size_i in sizes.items()
                  if(size_i<0.25) ]

def selection_exp(in_path,
                  n_splits=10,
                  n_repeats=1):
    data=dataset.read_csv(in_path)
    protocol=base.get_protocol("unaggr")(n_splits,n_repeats)
    splits=DataSplits( data=data,
                       splits=protocol.get_split(data))
    clf_factory=ClassEnsFactory(selected_classes=None)
    clfs=list(splits.get_clfs(clf_factory))
    for subset_i in iter_subsets(data):
        s_clfs=[SelectedEns(clf_j,subset_i) for clf_j in clfs]
        results=splits.pred(s_clfs)
        print(subset_i)
        print(np.mean(results.get_acc()))
        print(np.mean(results.get_balanced()))

def iter_subsets(data):
    cats= range(data.n_cats())
    for i in range(1,data.n_cats()):
        cats_i=itertools.combinations(cats, i)
        for cats_j in cats_i:
            yield cats_j
    yield list(cats)

selection_exp(in_path="../uci/wine-quality-red")
#clf.fit(data.X,data.y)
#clf.predict(data.X)
#model.summary()