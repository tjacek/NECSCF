import numpy as np
import tensorflow as tf
import base,dataset,deep,utils

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

ensemble_exp(in_path="../uci/wine-quality-red")
#clf.fit(data.X,data.y)
#clf.predict(data.X)
#model.summary()