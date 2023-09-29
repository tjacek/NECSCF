from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import data,train

def optim_rf(spilt):
    'max_depth': [10, 20] 


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
    print(type(spilt))