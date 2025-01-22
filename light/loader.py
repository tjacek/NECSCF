import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import argparse

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return int(max(self.y))+1

    def get_cat(self,i):
    	return self.X[self.y==i]
		
    def eval(self,train_index,test_index,clf):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history
        
    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def pred(self,test_index,clf):
        X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def class_percent(self):
        params,total_size={},float(len(self))
        for i in range(self.n_cats()):
            size_i= sum((self.y==i).astype(int))
            params[i]=size_i/total_size
        return params

    def selection(self,indices):
        return Dataset(X=self.X[indices],
                       y=self.y[indices])

class Split(object):
    def __init__(self,train_index,test_index):
            self.train_index=train_index
            self.test_index=test_index

        
    def eval(self,data,clf):
        return data.eval(train_index=self.train_index,
                             test_index=self.test_index,
                             clf=clf)
       
    def fit_clf(self,data,clf):
        return data.fit_clf(self.train_index,clf)

    def pred(self,data,clf):
        return data.pred(self.test_index,
                             clf=clf)

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

def simple_exp(data_path):
    data=read_csv(data_path)
    acc=[]
    for i,split_i in enumerate(get_splits(data)):
        clf_i=RandomForestClassifier()
        result_i,_=split_i.eval(data,clf_i)
        acc.append(result_i.get_acc())
        print(f"{i}:{acc[-1]:.4f}")
    print(f"Mean acc:{np.mean(acc):.4f}")
    print(f"Std acc:{np.std(acc):.4f}")

def get_splits(data,
               n_splits=10,
               n_repeats=10):
    rskf=RepeatedStratifiedKFold(n_repeats=n_repeats, 
                                 n_splits=n_splits, 
                                 random_state=0)
    splits=[]
    for train_index,test_index in rskf.split(data.X,data.y):
        yield Split(train_index,test_index)

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    if(type(in_path)!=str):
        return in_path
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../uci/vehicle")
    args = parser.parse_args()
    simple_exp(data_path=args.data_path)