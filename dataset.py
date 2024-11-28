import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,classification_report,balanced_accuracy_score

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
		
    def __call__(self,fun):
        return Dataset(X=fun(self.X),
                       y=self.y)
    
    def split(self,train_index,test_index):
        X_train=self.X[train_index]
        y_train=self.y[train_index]
        X_test=self.X[test_index]
        y_test=self.y[test_index]
        return (X_train,y_train),(X_test,y_test)        

    def eval(self,train_index,test_index,clf,as_result=True):
        (X_train,y_train),(X_test,y_test)=self.split(train_index,test_index)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        if(as_result):
            return Result(y_pred,y_test)
        else:
            return y_pred,y_test
            
    def labeled(self):
        return not (self.y is None)

    def min(self):
        return np.amin(self.X,axis=0)

    def max(self):
        return np.amax(self.X,axis=0)

    def class_percent(self):
        params,total_size={},float(len(self))
        for i in range(self.n_cats()):
            size_i= sum((self.y==i).astype(int))
            params[i]=size_i/total_size
        return params

    def selection(self,indices):
        return Dataset(X=self.X[indices],
                       y=self.y[indices])

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

    def get_balanced(self):
        return balanced_accuracy_score(self.y_pred,self.y_true)

    def get_metric(self,metric):
        return metric(self.y_pred,self.y_true)

    def report(self):
        print(classification_report(self.y_pred,self.y_true,digits=4))

    def true_pos(self):
        pos= [int(pred_i==true_i) 
                for pred_i,true_i in zip(self.y_pred,self.y_true)]
        return np.array(pos)

    def save(self,out_path):
        y_pair=np.array([self.y_pred,self.y_true])
        np.savez(out_path,y_pair)

class ResultGroup(object):
    def __init__(self,results):
        self.results=results

    def get_acc(self):
        return [result_j.get_acc() 
                    for result_j in self.results]

    def get_balanced(self):
        return [result_j.get_balanced() 
                    for result_j in self.results]

class WeightDict(dict):
    def __init__(self, arg=[]):
        super(WeightDict, self).__init__(arg)

    def Z(self):
        return sum(list(self.values()))

    def norm(self):
        Z=self.Z()
        for i in self:
            self[i]= self[i]/Z
        return self
    
    def size_dict(self):
        d={ i:(1.0/w_i) for i,w_i in self.items()}
        return  WeightDict(d).norm()

def read_result(in_path:str):
    if(type(in_path)==Result):
        return in_path
    raw=list(np.load(in_path).values())[0]
    y_pred,y_true=raw[0],raw[1]
    return Result(y_pred=y_pred,
                  y_true=y_true)

def compare_results(first_path,second_path):
    first,second=read_result(first_path),read_result(second_path)
    comp=first.true_pos() + (2*second.true_pos())
    return comp

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

def get_class_weights(y):
    params=WeightDict() 
    n_cats=int(max(y))+1
    for i in range(n_cats):
        size_i=sum((y==i).astype(int))
        if(size_i>0):
            params[i]= 1.0/size_i
        else:
            params[i]=0
    return params.norm()

def unify_results(partial_results):
    pairs=[ (result_i.y_pred,result_i.y_true) 
            for result_i in partial_results]
    y_pred,y_true=list(zip(*pairs))
    y_pred=np.concatenate(y_pred)
    y_true=np.concatenate(y_true)
    return Result(y_pred,y_true)

if __name__ == '__main__':
    data=read_csv("../uci/lymphography")
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        print(x_i.shape)