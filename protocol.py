import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import data

class Split(object):
    def __init__(self,X,y,train,valid,test):
        self.X=X 
        self.y=y 
        self.train=train
        self.valid=valid
        self.test=test
    
    def __str__(self):
    	return f'{len(self.train)},{len(self.valid)},{len(self.test)}'

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

if __name__ == '__main__':
    in_path='raw/mfeat-factors.arff'
    df=data.from_arff(in_path)
    X,y=data.prepare_data(df,target=-1)
    splits=list( gen_split(X,y))
    print(str(splits[0]))