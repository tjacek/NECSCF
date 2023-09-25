import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
import json
import protocol,utils

def pred_exp(in_path,out_path):
    metric=utils.get_metric('acc')
    utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        exp_i=protocol.read_exp(path_i)
        print(exp_i.params)
        y_true,y_pred= nn_exp(exp_i)
        print(f'acc:{metric(y_true,y_pred)}')
        name_i=path_i.split('/')[-1]
        out_i=f'{out_path}/{name_i}'
        np.savez(file=out_i,
                 true=y_true,
                 pred=y_pred)

def eval_exp(exp,verbose=0):
    extractor= exp.make_extractor()
    (x_train,y_train),(x_test,y_test)=exp.split.extract(extractor)
    clfs=[make_clf(x_i,y_train) 
            for x_i in x_train]
    votes=[clf_i.predict_proba(x_i)  
            for x_i,clf_i in zip(x_test,clfs)]
    y_pred=count_votes(votes)
    if(verbose):
        print(y_pred)
        print(y_test)
    return y_test,y_pred

def common_exp(exp):
    x_train,y_train=exp.split.get_train()
    x_test,y_test=exp.split.get_test()
    clf=make_clf(x_train,y_train)
    y_pred= clf.predict(x_test)
    return y_test,y_pred

def nn_exp(exp):
#    x_train,y_train=exp.get_train()
    x_test,y_test=exp.split.get_test()
    votes=exp.model.predict(x_test)  
    y_pred=count_votes(votes)
    print(y_pred)
    print(y_test)
    return y_test,y_pred

def count_votes(votes):
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def make_clf(X,y):
#    print(X.shape)
    clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
#    X=preprocessing.scale(X)
#    print(np.mean(X))
    clf.fit(X,y)
    return clf

def make_full(x,extractor):
    feats_train= extractor.predict(x)   
    return [ feats_i#np.concatenate([x,feats_i],axis=1) 
                for feats_i in feats_train]

if __name__ == '__main__':
    name='arrhythmia' #'vehicle'
    pred_exp(f'../OML/models/{name}',
             f'../OML/pred/{name}')

