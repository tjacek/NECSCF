import numpy as np
import tensorflow as tf
import dataset,deep

class DeepFactory(object):
    def __init__(self,hyper_params=None,selected_classes=None):
        if(hyper_params is None):
           hyper_params={'layers':2, 'units_0':2,
                         'units_1':1,'batch':False}
        self.params=None
        self.hyper_params=hyper_params
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'class_weights':dataset.get_class_weights(data.y) }

    def __call__(self):
        return Deep(params=self.params,
                    hyper_params=self.hyper_params)


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
        data=dataset.Dataset(X,y)
        if(self.model is None):
            self.model=deep.single_builder(params=self.params,
                                           hyper_params=self.hyper_params)
        y=tf.one_hot(y,depth=self.params['n_cats'])
        self.model.fit(x=X,
                       y=y,
                       callbacks=deep.get_callback(),
                       verbose=self.verbose)

    def predict(self,X):
        y=self.model.predict(X,
                             verbose=self.verbose)
        return np.argmax(y,axis=1)

class ClassEnsFactory(DeepFactory):

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
                                             hyper_params=self.hyper_params)
        y=[tf.one_hot(y,depth=self.params['n_cats'])
                for _ in range(data.n_cats()+1)]
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