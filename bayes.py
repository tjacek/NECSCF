from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import data,pred

class EvalBayes(object):
    def __init__(self):
        self.space=[#Integer(5,20,name='max_depth'),
                    Integer(10,100,name='n_estimators'),
                    Categorical(['balanced_subsample'],name='class_weight'),
                    Categorical(['gini','entropy'],name='criterion')]

    def __call__(self,exp,verbose=0):
        extractor= exp.make_extractor()
        data_tuples=exp.split.extract(extractor=extractor,
                                  use_valid=True)
        (x_train,y_train),(x_valid,y_valid),(x_test,y_test)=data_tuples
        @use_named_args(self.space)
        def objective(**params):
            print("call")
            y_pred=necscf(x_train,y_train,x_valid,params)
            return -accuracy_score(y_valid,y_pred)
        res_gp = gp_minimize(func=objective, 
                             dimensions=self.space, 
                             n_calls=50, 
                             random_state=0)            
        params=self.to_params(res_gp.x)
        print(params)
        y_pred=necscf(x_train,y_train,x_test,params)
        return y_test,y_pred
    
    def to_params(self,x):
        return {space_i.name:x_i 
                for space_i,x_i in zip(self.space,x)}

def necscf(x_train,y_train,x_test,params):
    votes=[]
    for train_i,test_i in zip(x_train,x_test):
            clf_i=ensemble.RandomForestClassifier(**params)
            clf_i.fit(train_i,y_train)
            votes.append(clf_i.predict_proba(test_i))
    y_pred=pred.count_votes(votes)
    return y_pred

def optim_rf(spilt,verbose=0):
    space=[Integer(5,20,name='max_depth'),
           Integer(5,20,name='n_estimators')] 
    x_train,y_train=spilt.get_train()
    x_valid,y_valid=spilt.get_valid()

    @use_named_args(space)
    def objective(**params):
        if(verbose):
            print(params)
        clf=ensemble.RandomForestClassifier(**params)#class_weight='balanced_subsample')
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_valid)
        return -accuracy_score(y_valid,y_pred)
    res_gp = gp_minimize(objective, 
                         space, 
                         n_calls=50, 
                         random_state=0)
    print(dir(res_gp))
    print(res_gp.x)
    print("Best score=%.4f" % res_gp.fun)
    return res_gp.x,res.fun

def eval_optim(spilt):
    x_train,y_train=spilt.get_train()
    x_valid,y_valid=spilt.get_valid()

    clf=ensemble.RandomForestClassifier()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_valid)
    print("Base score=%.4f" % accuracy_score(y_valid,y_pred))
    optim_rf(spilt)

if __name__ == '__main__':
    name='arrhythmia'
    in_path=f'raw/{name}.arff'
    target=-1
    df=data.from_arff(in_path)
    X,y=data.prepare_data(df,target=target)
    spilt=train.gen_split(X=X,
    	                  y=y,
    	                  n_iters=1)
    spilt=list(spilt)[0]
    eval_optim(spilt)