import numpy as np
import tensorflow as tf
import base,dataset,deep
import dataset,deep,ens_depen,utils

def get_ens(ens_type:str,hyper_params=None):
    if(ens_type=="MLP"):
        return DeepFactory(hyper_params)
    if(ens_type=="class_ens"):
        return MultiEnsFactory(hyper_params=None,
                               loss_gen=deep.WeightedLoss())
    if(ens_type=="purity_ens"):
        return MultiEnsFactory(hyper_params=hyper_params,
                               loss_gen=ens_depen.PurityLoss())
    if(ens_type=="separ_class_ens"):
        loss_gen=deep.WeightedLoss(multi=False)
        return SeparatedEnsFactory(hyper_params=None,
                                   loss_gen=loss_gen)
    if(ens_type=="separ_purity_ens"):
        loss_gen=ens_depen.PurityLoss(multi=False)
        return SeparatedEnsFactory(hyper_params=None,
                                   loss_gen=loss_gen)
    if(ens_type=="deep"):
        return DeepFactory()
    if(ens_type=="RF" or ens_type=="GRAD"):
        return ClasicalClfFactory(ens_type)
    raise Exception(f"Unknown ens type:{ens_type}")

def is_neural(ens_type):
    return ("ens" in ens_type or "deep"==ens_type)

def is_ensemble(ens_type):
    return "ens" in ens_type

def is_separ(ens_type):
    return "separ" in ens_type

def default_hyperparams():
    return {'layers':2, 'units_0':2,
            'units_1':1,'batch':False}

class ClfFactory(object):
    def __init__(self,hyper_params=None,
                      loss_gen=None):
        if(hyper_params is None):
           hyper_params=default_hyperparams()
        self.params=None
        self.hyper_params=hyper_params
        self.class_dict=None
        self.loss_gen=loss_gen
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000}
        self.class_dict=dataset.get_class_weights(data.y)

    def __call__(self):
        raise NotImplementedError()

    def read(self,model_path):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

class ClfAdapter(object):
    def __init__(self, params,
                       hyper_params,
                       class_dict=None,
                       model=None,
                       loss_gen=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.class_dict=class_dict
        self.model = model
        self.loss_gen=loss_gen
        self.verbose=verbose

    def fit(self,X,y):
        raise NotImplementedError()

    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.partial_predict(test_data_i.X)
        result_i=dataset.PartialResults(y_true=test_data_i.y,
                                        y_partial=raw_partial_i)
        return result_i

    def save(self,out_path):
        raise NotImplementedError()

class DeepFactory(ClfFactory):
    def __call__(self):
        return Deep(params=self.params,
                    hyper_params=self.hyper_params,
                    class_dict=self.class_dict)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path,
                                           custom_objects={"loss":deep.WeightedLoss})
        clf_i=self()
        clf_i.model=model_i
        return clf_i#Deep(clf_i,hyper_params=self.hyper_params,)

    def get_info(self):
        return {"ens":"deep","callback":"total","hyper":self.hyper_params}

class Deep(ClfAdapter):

    def fit(self,X,y):
        if(self.model is None):
            self.model=deep.single_builder(params=self.params,
                                           hyper_params=self.hyper_params,
                                           class_dict=self.class_dict)
        y=tf.one_hot(y,depth=self.params['n_cats'])
        return self.model.fit(x=X,
                              y=y,
                              epochs=self.params['n_epochs'],
                              callbacks=ens_depen.basic_callback(),
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

    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.predict(test_data_i.X)
        result_i=dataset.Result(y_true=test_data_i.y,
                                y_pred=raw_partial_i)
        return result_i


class ClasicalClfFactory(ClfFactory):
    def __init__(self,clf_type="RF"):
        self.clf_type=clf_type
    
    def init(self,data):
        pass

    def __call__(self):
        return ClasicalClfAdapter(base.get_clf(self.clf_type))

    def read(self,model_path):
        raise Exception(f"Clasical Clf {self.clf_type} cannot be serialized ")

    def get_info(self):
        return {"ens":self.clf_type,"callback":None,"hyper":None}

class ClasicalClfAdapter(ClfAdapter):
    def __init__(self,clf):
        self.clf=clf
    
    def fit(self,X,y):
        return self.clf.fit(X,y)

    def eval(self,data,split_i):
        return split_i.pred(data,self.clf)

    def save(self,out_path):
        pass

class MultiEnsFactory(ClfFactory):

    def __call__(self):
        return MultiEns(params=self.params,
                        hyper_params=self.hyper_params,
                        class_dict=self.class_dict,
                        loss_gen=self.loss_gen)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path,
                                           custom_objects={"loss":self.loss_gen})
        clf_i=self()
        clf_i.model=model_i
        return clf_i

    def get_info(self):
        return {"ens":str(self.loss_gen),"callback":"total","hyper":self.hyper_params}

class MultiEns(ClfAdapter):

    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        self.loss_gen.init(data)
        if(self.model is None):

            self.model=deep.ensemble_builder(params=self.params,
                                             hyper_params=self.hyper_params,
                                             loss_gen=self.loss_gen,
                                             class_dict=self.class_dict)
        y=[tf.one_hot(y,depth=self.params['n_cats'])
                for _ in range(data.n_cats()+1)]
        callback="total"#self.hyper_params["callback"]
        if(type(callback)==str):
            callback=ens_depen.get_callback(callback)(verbose=0)
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

class SeparatedEnsFactory(ClfFactory):
    def __call__(self):
        return SeparatedEns(params=self.params,
                        hyper_params=self.hyper_params,
                        loss_gen=self.loss_gen)
    
    def __call__(self):
        return SeparatedEns(params=self.params,
                            hyper_params=self.hyper_params,
                            class_dict=self.class_dict,
                            loss_gen=self.loss_gen)

    def read(self,model_path):
        clf_i=self()
        clf_i.model=[]
        for path_j in utils.top_files(model_path):
            deep_i=Deep(params=self.params,
                        hyper_params=self.hyper_params)
            deep_i.model=tf.keras.models.load_model(path_j,
                                                    custom_objects={"loss":self.loss_gen})
            clf_i.model.append(deep_i)
        return clf_i

    def get_info(self):
        return {"ens":str(self.loss_gen),"callback":"total","hyper":self.hyper_params}

class SeparatedEns(ClfAdapter):
    def __init__(self,*args, **kwargs):
        super(SeparatedEns, self).__init__(*args, **kwargs)
        self.model=[]
    
    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        self.loss_gen.init(data)
        n_cats=self.params["n_cats"]
        history=[]
        indexes= list(range(n_cats))+[None]
        for i in indexes:#range(n_cats):
            class_dict_i=self.loss_gen(i,self.class_dict)
            model_i=Deep(params=self.params,
                         hyper_params=self.hyper_params,
                         class_dict=class_dict_i)
            self.model.append(model_i)
            history_i=model_i.fit(X,y)
            history.append(history_i)

        return history

    def partial_predict(self,X):
        votes=[clf_i.predict_proba(X) for clf_i in self.model]
        return np.array(votes)    

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,clf_i in enumerate(self.model):
            clf_i.save(f"{out_path}/{i}.keras")
