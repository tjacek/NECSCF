import numpy as np
from sklearn import ensemble
import protocol,utils

def eval_exp(exp):
    x_train,y_train=exp.get_train()
    x_test,y_test=exp.get_test()
    extractor= exp.make_extractor()
    feats_train= extractor.predict(x_train)   
    full_train=[np.concatenate([x_train,feats_i],axis=1) 
            for feats_i in feats_train]
    clfs=[make_clf(full_i,y_train) for full_i in full_train]
    feats_test= extractor.predict(x_test)   
    full_test=[np.concatenate([x_test,feats_i],axis=1) 
                for feats_i in feats_test]
    votes=[clf_i.predict_proba(full_i)  
               for full_i,clf_i in zip(full_test,clfs)]
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    y_pred=np.argmax(votes,axis=1)
    return y_test,y_pred

def make_clf(X,y):
	print(len(y))
	clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
	clf.fit(X,y)
	return clf

if __name__ == '__main__':
    exp=protocol.read_exp('out_h5/1')
    y_test,y_pred= eval_exp(exp)
    metric=utils.get_metric('acc')
    print(metric(y_test,y_pred))