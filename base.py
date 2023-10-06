import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import preprocessing
from collections import defaultdict
import json,random
import utils,deep,data

class AlgParams(obejct):
    def __init__(self,hyper_type='eff',epochs=300,callbacks=None):
        if(callbacks is None):
            callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=5)
        self.hyper_type=hyper_type
        self.epochs=epochs
        self.callbacks=callbacks

class Experiment(object):
    def __init__(self,split,params,hyper_params=None,model=None):
        self.split=split
        self.params=params
        self.hyper_params=hyper_params
        self.model=model

    def eval(self,clf_type="RF"):
        extractor=self.make_extractor() 
        train,valid,test=self.split.extract(extractor=extractor,
                                            use_valid=True)
        x_train,y_train=train
        x_valid,y_valid=valid
        y_pred=simple_necscf(x_train=x_train,
                             y_train=y_train,
                             x_test=x_valid,
                             clf_type=clf_type)
        accuracy=utils.get_metric('acc')
        return accuracy(y_pred,y_valid)

    def train(self,alg_params,verbose=0,alpha=0.5):
        if(self.model is None):
            self.model=deep.ensemble_builder(params=self.params,
                                             hyper_params=self.hyper_params,
                                             alpha=alpha)
            if(verbose):
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
                       epochs=alg_params.epochs,
                       validation_data=(x_valid, y_valid),
                       verbose=verbose,
                       callbacks=alg_params.callbacks)

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
        if(not (self.model is None)):
            self.model.save(f'{out_path}/nn')

def read_exp(in_path):
    split=read_split(in_path)
    params=data.get_dataset_params(X=split.X,
                                   y=split.y)
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

#def gen_split(X,y,n_iters=2):
#    for i in range(n_iters):
#        rkf = RepeatedStratifiedKFold(n_splits=5, 
#    	                              n_repeats=1, 
#    	                              random_state=42)
#        k_folds=[test_idx  for train_idx, test_idx in rkf.split(X, y)]
#        valid=k_folds[0]
#        test=k_folds[1]
#        train=np.concatenate(k_folds[2:])
#        yield Split(X=X,
#                    y=y,
#                    train=train,
#        	        valid=valid,
#        	        test=test)

def gen_split(X,y,n_iters=2):
    for i in range(n_iters):
        train_i,valid_i,test_i=split_data(X,y)
        yield Split(X=X,
                    y=y,
                    train=train_i,
                    valid=valid_i,
                    test=test_i)

def split_data(X,y):
    by_cat=defaultdict(lambda :[])
    for i,cat_i in enumerate( y):
        by_cat[cat_i].append(i)
    train,valid,test=[],[],[]
    for cat_i,samples_i in by_cat.items():
        random.shuffle(samples_i)
        for j,index in enumerate(samples_i):
            mod_j= j % 5
            if(mod_j==4):
                valid.append(index)
            elif(mod_j==0):
                test.append(index)
            else:
                train.append(index)
    return train,valid,test   

def single_split(X,y):
    return list(gen_split(X,y))[0]

def read_split(in_path):
    dataset=np.load(f'{in_path}/data.npz')
    X,y=dataset['X'],dataset['y']
    split_raw=np.load(f'{in_path}/split.npz')
    return Split(X=X,
                y=y,
                train=split_raw['train'],
                valid=split_raw['valid'],
                test=split_raw['test'])

def simple_necscf(x_train,y_train,x_test,clf_type):
    votes=[]
    for train_i,test_i in zip(x_train,x_test):
            clf_i=get_clf(clf_type)#ensemble.RandomForestClassifier(**params)
            clf_i.fit(train_i,y_train)
            votes.append(clf_i.predict_proba(test_i))
    y_pred=count_votes(votes)
    return y_pred

def count_votes(votes):
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)