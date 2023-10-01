import utils
utils.silence_warnings()
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import Input, Model
import json
import data,deep,hyper

class Experiment(object):
    def __init__(self,split,params,hyper_params=None,model=None):

        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model

    def find_hyper(self,n_iter=5,verbose=1):
        self.hyper_params=hyper.bayes_optim(exp=self,
                                            n_iter=n_iter,
                                            verbose=verbose)

    def train(self,epochs=300,verbose=0,callbacks=None):
        if(self.model is None):
            self.model=deep.ensemble_builder(params=self.params,
                                             hyper_params=self.hyper_params)
            self.model.summary()
        x_train,y_train=self.split.get_train()
        y_train=[tf.keras.utils.to_categorical(y_train) 
                    for k in range(self.params['n_cats'])]
        x_valid,y_valid=self.split.get_valid()
        y_valid=[tf.keras.utils.to_categorical(y_valid) 
                    for k in range(self.params['n_cats'])]
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=self.params['batch'],
                       epochs=epochs,
                       validation_data=(x_valid, y_valid),
                       verbose=verbose,
                       callbacks=callbacks)

    def make_extractor(self):
        names= [ layer.name for layer in self.model.layers]
        n_cats=self.params['n_cats']
        penult=names[-2*n_cats:-n_cats]
        layers=[self.model.get_layer(name_i).output 
                    for name_i in penult]
        return Model(inputs=self.model.input,
                        outputs=layers)

    def save(self,out_path):
        utils.make_dir(out_path)
        self.split.save(f'{out_path}/split')
        np.savez(file=f'{out_path}/data',
                 X=self.split.X,
                 y=self.split.y)
        with open(f'{out_path}/hyper.json', 'w') as f:
            json.dump(self.hyper_params, f)
        self.model.save(f'{out_path}/nn')

def read_exp(in_path):
    dataset=np.load(f'{in_path}/data.npz')
    X,y=dataset['X'],dataset['y']
    split_raw=np.load(f'{in_path}/split.npz')
    split=Split(X=X,
                y=y,
                train=split_raw['train'],
                valid=split_raw['valid'],
                test=split_raw['test'])
    params=data.get_dataset_params(X,y)
    class_dict=params['class_weights']
    with open(f'{in_path}/hyper.json', 'r') as f:        
        json_bytes = f.read()                      
        hyper_params=json.loads(json_bytes)
    model = tf.keras.models.load_model(f'{in_path}/nn',
                                         compile=False)
    return Experiment(split=split,
                      params=params,
                      hyper_params=hyper_params,
                      model=model)

class Split(object):
    def __init__(self,X,y,train,valid,test):
        self.X=X
        self.y=y
        self.train=train
        self.valid=valid
        self.test=test
    
    def get_train(self):
        return self.X[self.train],self.y[self.train]
    
    def get_valid(self):
        return self.X[self.valid],self.y[self.valid]

    def get_test(self):
        return self.X[self.test],self.y[self.test]
 
    def extract(self,extractor,use_valid=False):
        cs=extractor.predict(self.X)
        train,test,valid=[],[],[]
        for cs_i in cs:
            cs_i= preprocessing.RobustScaler().fit_transform(cs_i)
            full_i=np.concatenate([self.X,cs_i],axis=1)
            train.append(full_i[self.train])
            test.append(full_i[self.test])
            if(use_valid):
                valid.append(full_i[self.valid])
        train_tuple= (train,self.y[self.train])
        test_tuple = (test,self.y[self.test])
        if(use_valid):
            valid_tuple=(valid,self.y[self.valid])
            return train_tuple,valid_tuple,test_tuple
        return train_tuple,test_tuple

    def save(self,out_path):
        np.savez(file=out_path,
                 train=self.train,
                 valid=self.valid,
                 test=self.test )
    
    def __str__(self):
    	return f'{len(self.train)},{len(self.valid)},{len(self.test)}'

    def eval(self,clf_type):
        clf=get_clf(clf_type)
        x_train,y_train=self.get_train()
        clf.fit(x_train,y_train)
        x_test,y_test=self.get_test()
        return y_test,clf.predict(x_test)

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="SVC"):
        return SVC(probability=True)
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')

def gen_split(X,y,n_iters=2):
    for i in range(n_iters):
        rkf = RepeatedStratifiedKFold(n_splits=5, 
    	                              n_repeats=1, 
    	                              random_state=42)
        k_folds=[test_idx  for train_idx, test_idx in rkf.split(X, y)]
        valid=k_folds[0]
        test=k_folds[1]
        train=np.concatenate(k_folds[2:])
        yield Split(X=X,
                    y=y,
                    train=train,
        	        valid=valid,
        	        test=test)

def all_train(in_path,out_path,n_iters=2):
    names={'arrhythmia':-1}#,'mfeat-factors':-1,'vehicle':-1,
#           'cnae-9':-1,'car':-1,'segment':-1,'fabert':0}
    for name_i,target_i in names.items():
        train_exp(in_path=f'{in_path}/{name_i}.arff',
                  out_path=f'{out_path}/{name_i}',
                  n_iters=n_iters,
                  target=-1 )

@utils.log_time(task='TRAIN')
def train_exp(in_path,out_path,n_iters=2,hyper=None,target=-1 ):
    df=data.from_arff(in_path)
    X,y=data.prepare_data(df,target=target)
    params=data.get_dataset_params(X,y)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=5)
    utils.make_dir(out_path)
    for i,split_i in enumerate(gen_split(X,y,n_iters)):
        exp_i=Experiment(split=split_i,
        	             params=params)	
        exp_i.find_hyper()
        exp_i.train(verbose=1,
                    callbacks=stop_early)
        exp_i.save(f'{out_path}/{i}')

if __name__ == '__main__':
#    name='arrhythmia'#cnae-9'
#    in_path=f'raw/{name}.arff'
    utils.start_log('log.info')
    all_train(in_path='raw', #'../OML/models',
              out_path='../OML/_models',
              n_iters=2)
#    train_exp(in_path=in_path,
#    	      out_path=f'../OML/models/{name}',
#              n_iters=10,
#              target=-1)