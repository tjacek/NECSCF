import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
import json
import protocol,utils

class OptimRF(object):
    def __init__(self, arg):
        self.clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
        self.search_space={'max_samples':(0.5,1),
                           'max_features':(0.5,1),
                           'n_estimators':(100,200)
                          }

    def find_params(self,split):
        search = BayesSearchCV(estimator=clf,
                               verbose=0,
                               n_iter=5,
                               search_spaces=self.search_spaces,
                               n_jobs=1,cv=cv_gen,
                               scoring=self.scoring)

#    def fit(self,X,y):

def all_pred(in_path,out_path):
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        pred_exp(in_path=f'{path_i}',
                 out_path=f'{out_path}/{name_i}')

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

#def make_optim_clf(split):
#    x_train,y_train=split.get_train()
#    rf_opt = BayesianOptimization(bo_params_rf,)

def make_clf(split):#X,y):
    X,y=split.get_train()
    clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    clf.fit(X,y)
    return clf

if __name__ == '__main__':
#    name='arrhythmia' #'vehicle'
#    pred_exp(f'../OML/models/{name}',
#             f'../OML/pred/{name}')
    all_pred(f'../OML/models',
             f'../OML/pred')
