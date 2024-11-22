import numpy as np
import tensorflow as tf
import dataset,deep

class NECSCF(object):
    def __init__(self, model=None):
        self.model = model
		
    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        if(self.model is None):
            params={'dims': (data.dim(),),
                    'n_cats':data.n_cats(),
                    'class_weights':dataset.get_class_weights(data.y) }
            self.model=deep.ensemble_builder(params,
	                                         hyper_params=None,
	                                         alpha=0.5)
        y=[tf.one_hot(y,
                      depth=params['n_cats'])
                for i in range(data.n_cats())]

        self.model.fit(x=X,
        	           y=y,
        	           callbacks=deep.get_callback())

    def predict(self,X):
    	y=self.model.predict(X)
    	y=np.sum(np.array(y),axis=2)
    	return np.argmax(y,axis=0)

data=dataset.read_csv("../uci/cmc")
clf=NECSCF()
clf.fit(data.X,data.y)
clf.predict(data.X)
#model.summary()