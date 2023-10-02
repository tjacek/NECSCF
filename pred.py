import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
import json
import base,train,utils,bayes

def all_pred(in_path,out_path):
    metric=utils.get_metric('acc')
    @utils.dir_fun
    def helper(in_path,out_path):
        exp_i=base.read_exp(in_path)
        y_true,y_pred=nescf_eval(exp_i)
        print(f'{out_path} Acc:{metric(y_true,y_pred):.4f}')
        np.savez(file=out_path,
                 true=y_true,
                 pred=y_pred)
    helper(in_path,out_path)
#    utils.make_dir(out_path)
#    for path_i in utils.top_files(in_path):
#        name_i=path_i.split('/')[-1]
#        pred_exp(in_path=f'{path_i}',
#                 out_path=f'{out_path}/{name_i}',
#                 eval=pred_exp)#bayes.EvalBayes())

def nescf_eval(exp):
    extractor=exp.make_extractor() 
    train,test=exp.split.extract(extractor=extractor,
                                 use_valid=False)
    (x_train,y_train),(x_test,y_test)=train,test
    y_pred=base.simple_necscf(x_train=x_train,
                              y_train=y_train,
                              x_test=x_test,
                              clf_type="RF")
    return y_test,y_pred
#def pred_exp(in_path,out_path,eval_exp):
#    metric=utils.get_metric('acc')
#    utils.make_dir(out_path)
#    for path_i in utils.top_files(in_path):
#        exp_i=base.read_exp(path_i)
#        print(exp_i.params)
#        y_true,y_pred= eval_exp(exp_i)
#        print(f'acc:{metric(y_true,y_pred):.4f}')
#        name_i=path_i.split('/')[-1]
#        out_i=f'{out_path}/{name_i}'
#        np.savez(file=out_i,
#                 true=y_true,
#                 pred=y_pred)


#def eval_exp(exp,verbose=0):
#    extractor= exp.make_extractor()
#    (x_train,y_train),(x_test,y_test)=exp.split.extract(extractor)
#    clfs=[make_clf(x_i,y_train) 
#            for x_i in x_train]
#    votes=[clf_i.predict_proba(x_i)  
#            for x_i,clf_i in zip(x_test,clfs)]
#    y_pred=count_votes(votes)
#    if(verbose):
#        print(y_pred)
#        print(y_test)
#    return y_test,y_pred

def common_exp(exp):
    x_train,y_train=exp.split.get_train()
    x_test,y_test=exp.split.get_test()
    clf=make_clf(x_train,y_train)
    y_pred= clf.predict(x_test)
    return y_test,y_pred

def nn_exp(exp):
    x_test,y_test=exp.split.get_test()
    votes=exp.model.predict(x_test)  
    y_pred=count_votes(votes)
    return y_test,y_pred

def make_clf(split):#X,y):
    X,y=split.get_train()
    clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    clf.fit(X,y)
    return clf

if __name__ == '__main__':
#    name='arrhythmia' #'vehicle'
#    pred_exp(f'../OML/models/{name}',
#             f'../OML/pred/{name}')
    all_pred(f'../OML_reduce/models',
             f'../OML_reduce/pred')
