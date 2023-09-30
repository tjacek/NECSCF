from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import data,train

def optim_rf(spilt):
    space=[Integer(5,20,name='max_depth'),
           Integer(5,20,name='n_estimators')] 
    x_train,y_train=spilt.get_train()
    x_valid,y_valid=spilt.get_valid()

    @use_named_args(space)
    def objective(**params):
#         clf.set_params(**params)
        print(params)
        clf=ensemble.RandomForestClassifier(**params)#class_weight='balanced_subsample')
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_valid)
        return accuracy_score(y_valid,y_pred)
    res_gp = gp_minimize(objective, 
                         space, 
                         n_calls=50, 
                         random_state=0)
    print("Best score=%.4f" % res_gp.fun)
         
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
    optim_rf(spilt)