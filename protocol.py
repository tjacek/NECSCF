import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
import json
import data,deep,hyper,utils

class Experiment(object):
    def __init__(self,X,y,split,params,hyper_params=None,model=None):
        self.X=X
        self.y=y
        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model
    
    def get_train(self):
        split_i=self.split.train
        return self.X[split_i],self.y[split_i]
    
    def get_valid(self):
        split_i=self.split.valid
        return self.X[split_i],self.y[split_i]

    def find_hyper(self,n_iter=5,verbose=1):
        self.hyper_params=hyper.bayes_optim(exp=self,
                                            n_iter=n_iter,
                                            verbose=verbose)

    def train(self,epochs=150,verbose=0,callbacks=None):
        if(self.model is None):
            self.model=deep.ensemble_builder(params=self.params,
                                             hyper_params=self.hyper_params)
            self.model.summary()
        x_train,y_train=self.get_train()
        y_train=[tf.keras.utils.to_categorical(y_train) 
                    for k in range(self.params['n_cats'])]
        x_valid,y_valid=self.get_valid()
        y_valid=[tf.keras.utils.to_categorical(y_valid) 
                    for k in range(self.params['n_cats'])]
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=self.params['batch'],
                       epochs=epochs,
                       validation_data=(x_valid, y_valid),
                       verbose=verbose,
                       callbacks=callbacks)

    def save(self,out_path):
        utils.make_dir(out_path)
        self.split.save(f'{out_path}/split')
        np.savez(file=f'{out_path}/data',
                 X=self.X,
                 y=self.y)
        with open(f'{out_path}/hyper.json', 'w') as f:
            json.dump(self.hyper_params, f)
        self.model.save(f'{out_path}/nn')

def read_exp(in_path):
    split_raw=np.load(f'{in_path}/split.npz')
    split=Split(train=split_raw['train'],
                valid=split_raw['valid'],
                test=split_raw['test'])
    dataset=np.load(f'{in_path}/data.npz')
    X,y=dataset['X'],dataset['y']
    params=data.get_dataset_params(X,y)
    class_dict=params['class_weights']
    with open(f'{in_path}/hyper.json', 'r') as f:        
        json_bytes = f.read()                      
        hyper_params=json.loads(json_bytes)
    model = tf.keras.models.load_model(f'{in_path}/nn',
                                         compile=False)
    return Experiment(X=X,
                      y=y,
                      split=split,
                      params=params,
                      hyper_params=hyper_params,
                      model=model)

class Split(object):
    def __init__(self,train,valid,test):
        self.train=train
        self.valid=valid
        self.test=test

    def save(self,out_path):
        np.savez(file=out_path,
                 train=self.train,
                 valid=self.valid,
                 test=self.test )
    
    def __str__(self):
    	return f'{len(self.train)},{len(self.valid)},{len(self.test)}'

def gen_split(X,y,n_iters=2):
    for i in range(n_iters):
        rkf = RepeatedStratifiedKFold(n_splits=5, 
    	                              n_repeats=1, 
    	                              random_state=42)
        k_folds=[test_idx  for train_idx, test_idx in rkf.split(X, y)]
        valid=k_folds[0]
        test=k_folds[1]
        train=np.concatenate(k_folds[2:])
        yield Split(train=train,
        	        valid=valid,
        	        test=test)

def train_exp(in_path,out_path,n_iters=2,hyper=None):
    if(hyper is None):
    	hyper={'layers':[150,150],'batch':True}
    df=data.from_arff(in_path)
    X,y=data.prepare_data(df,target=-1)
    params=data.get_dataset_params(X,y)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5)
    utils.make_dir(out_path)
    for i,split_i in enumerate(gen_split(X,y,n_iters)):
        exp_i=Experiment(X=X,
                         y=y,
        	             split=split_i,
        	             params=params)	
        exp_i.find_hyper()
        exp_i.train(verbose=1,
                    callbacks=stop_early)
        exp_i.save(f'{out_path}/{i}')

if __name__ == '__main__':
    in_path='raw/mfeat-factors.arff'
    train_exp(in_path=in_path,
    	      out_path='out',
              n_iters=2)