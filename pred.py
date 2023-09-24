import numpy as np
from sklearn import ensemble
import json
import protocol,utils

def pred_exp(in_path,out_path):
    metric=utils.get_metric('acc')
    utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        exp_i=protocol.read_exp(path_i)
        y_true,y_pred= eval_exp(exp_i)
        print(metric(y_true,y_pred))
        name_i=path_i.split('/')[-1]
        out_i=f'{out_path}/{name_i}'
        np.savez(file=out_i,
                 true=y_true,
                 pred=y_pred)

def eval_exp(exp):
    x_train,y_train=exp.get_train()
    x_test,y_test=exp.get_test()
    extractor= exp.make_extractor()
    full_train=make_full(x_train,extractor)
    clfs=[make_clf(full_i,y_train) 
            for full_i in full_train]
    full_test=make_full(x_test,extractor)   
    votes=[clf_i.predict_proba(full_i)  
            for full_i,clf_i in zip(full_test,clfs)]
    y_pred=count_votes(votes)
    return y_test,y_pred

def count_votes(votes):
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def make_clf(X,y):
	clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
	clf.fit(X,y)
	return clf

def make_full(x,extractor):
    feats_train= extractor.predict(x)   
    return [np.concatenate([x,feats_i],axis=1) 
                for feats_i in feats_train]

if __name__ == '__main__':
    pred_exp('out_h5','pred')

#    print(metric(y_test,y_pred))