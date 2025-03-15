import numpy as np
import tensorflow as tf
import base,dataset,deep
import ens_dep 

def get_ens(ens_type:str,hyper_params=None):
    if(ens_type=="MLP"):
        return DeepFactory(hyper_params)
    if(ens_type=="class_ens"):
        return ClassEnsFactory(hyper_params)
    if(ens_type=="purity_ens"):
        return ClassEnsFactory(hyper_params=hyper_params,
                               loss_gen=ens_deep.PurityLoss())
    if(ens_type=="RF"):
        return base.ClfFactory(ens_type)
    raise Exception(f"Unknow ens type{ens_type}")

def default_hyperparams():
    return {'layers':2, 'units_0':2,
            'units_1':1,'batch':False,
            'callback':'total'}

class DeepFactory(object):
    def __init__(self,hyper_params=None):
        if(hyper_params is None):
           hyper_params=default_hyperparams()
        self.params=None
        self.hyper_params=hyper_params
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000,
                     'class_weights':dataset.get_class_weights(data.y) }

    def __call__(self):
        return Deep(params=self.params,
                    hyper_params=self.hyper_params)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path,
                                           custom_objects={"loss":deep.WeightedLoss})
        clf_i=self()
        clf_i.model=model_i
        return clf_i

class Deep(object):
    def __init__(self, params,
                       hyper_params,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.model = model
        self.verbose=verbose

    def fit(self,X,y):
        if(self.model is None):
            self.model=deep.single_builder(params=self.params,
                                           hyper_params=self.hyper_params)
        y=tf.one_hot(y,depth=self.params['n_cats'])
        return self.model.fit(x=X,
                              y=y,
                              epochs=self.params['n_epochs'],
                              callbacks=deep.basic_callback(),
                              verbose=self.verbose)

    def predict(self,X):
        y=self.model.predict(X,
                             verbose=self.verbose)
        return np.argmax(y,axis=1)

    def predict_proba(self,X):
        return self.model.predict(X,
                             verbose=self.verbose)

    def save(self,out_path):
        self.model.save(out_path) 

class ClassEnsFactory(DeepFactory):
    def __init__(self,loss_gen=None,
                      hyper_params=None):
        if(loss_gen is None):
            loss_gen=deep.WeightedLoss()
        if(hyper_params is None):
           hyper_params=default_hyperparams()
        self.params=None
        self.loss_gen=loss_gen
        self.hyper_params=hyper_params
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000,
                     'class_weights':dataset.get_class_weights(data.y) }

    def __call__(self):
        return ClassEns(params=self.params,
                        hyper_params=self.hyper_params,
                        loss_gen=self.loss_gen)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path,
                                           custom_objects={"loss":self.loss_gen})
        clf_i=self()
        clf_i.model=model_i
        return clf_i

class ClassEns(object):
    def __init__(self, params,
                       hyper_params,
                       loss_gen,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.loss_gen=loss_gen
        self.model = model
        self.verbose=verbose

    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        self.loss_gen.init(data)
        if(self.model is None):
            self.model=deep.ensemble_builder(params=self.params,
                                             hyper_params=self.hyper_params,
                                             loss_gen=self.loss_gen)
        y=[tf.one_hot(y,depth=self.params['n_cats'])
                for _ in range(data.n_cats()+1)]
        callback=self.hyper_params["callback"]
        if(type(callback)==str):
            callback=deep.get_callback(callback)(verbose=0)
        callback.init(data.n_cats()+1)
        return self.model.fit(x=X,
                       y=y,
                       epochs=self.params['n_epochs'],
                       batch_size=self.params['dims'][0],
                       callbacks=callback,
                       verbose=self.verbose)

    def predict(self,X):
        y=self.model(X, training=False)
        y=np.sum(np.array(y),axis=0)
        return np.argmax(y,axis=1)

    def partial_predict(self,X):
        return np.array(self.model(X, training=False))
    
    def select_predict(self,X,select_cats):
        y=self.model.predict(X,
                             verbose=self.verbose)
        y=[y[cat_i] for cat_i in select_cats]
        y=np.sum(np.array(y),axis=0)
        return np.argmax(y,axis=1)

    def save(self,out_path):
        self.model.save(out_path) 


class NECSCF(object):
    def __init__(self,model):
        self.model=all_splits
        self.clfs=[]

    def fit(X,y,clf_type="RF"):
        feats=self.get_features(X)
        self.clfs=[]
        for feat_i in feats:
            clf_i=base.get_clf(clf_type)
            clf_i.fit(feat_i,y)
            self.clfs.append(clf_i)

    def pred(self,X):
        feats=self.get_features(X)
        votes=[clf_i.pred(feat_i) for feat_i,clf_i in zip(feats,self.clfs)]
        y=np.sum(np.array(y),axis=0)
        return np.argmax(y,axis=1)