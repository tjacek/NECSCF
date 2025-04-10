import pandas as pd
import numpy as np
#from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
from sklearn.metrics import classification_report
import utils

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
                           
    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf,as_result=as_result)
        return result,history
        
    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def pred(self,test_index,clf,as_result=True):
        X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        if(as_result):
            return Result(y_pred,y_test)
        else:
            return y_pred,y_test
    
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

    def get_metric(self,metric_type):
        if(type(metric_type)==str):
            metric=dispatch_metric(metric_type)
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

    def get_metric(self,metric_type):
        return [result_j.get_metric(metric_type) 
                    for result_j in self.results]
    def get_acc(self):
        return [result_j.get_acc() 
                    for result_j in self.results]

    def get_balanced(self):
        return [result_j.get_balanced() 
                    for result_j in self.results]

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,result_i in enumerate(self.results):
            result_i.save(f"{out_path}/{i}")

class PartialResults(object):
    def __init__(self,y_true,y_partial):
        self.y_true=y_true
        self.y_partial=y_partial

    def __len__(self):
        return self.y_partial.shape[0]

    def vote(self):
        ballot= np.sum(self.y_partial,axis=0)
        return np.argmax(ballot,axis=1)

    def get_metric(self,metric_type="acc"):
        y_pred=self.vote()
        metric=dispatch_metric(metric_type)
        return metric(self.y_true,y_pred)
    
    def selected_acc(self,subset,metric_type="acc"):
        s_votes=[self.y_partial[i] for i in subset]
        s_ballot= np.sum(s_votes,axis=0)
        s_pred=np.argmax(s_ballot,axis=1)
        metric=dispatch_metric(metric_type)
        return metric(self.y_true,s_pred)

    def save(self,out_path):
        np.savez(out_path,name1=self.y_partial,name2=self.y_true)

    def __str__(self):
        return str(self.y_true.shape)

class PartialGroup(object):
    def __init__(self,partials):
        self.partials=partials
   
    def n_clfs(self):
        return self.partials[0].y_partial.shape[0]

    def get_metric(self,metric_type="acc",subset=None):
        if(subset is None):
            return [result_j.get_metric(metric_type) 
                        for result_j in self.partials]
        else:
            return [result_j.selected_acc(metric_type=metric_type,
                                          subset=subset) 
                        for result_j in self.partials]

    def order_acc(self,order_i,metric_type="acc",full=True):
        subsets=utils.selected_subsets(order_i,full=full)
        acc=[self.get_metric(metric_type=metric_type,
                             subset=subset_j) 
                for subset_j in subsets]
        return np.array(acc)

    def indv_acc(self,metric_type="acc"):
        n_clf= self.n_clfs()
        return [ self.get_metric([i],metric_type) 
                    for i in range(n_clf)]

def dispatch_metric(metric_type):
    if(metric_type=="acc"):
        return accuracy_score
    if(metric_type=="balance"):
        return balanced_accuracy_score
    if(metric_type=="f1-score"):
        return f1_score
    raise Exception(f"Unknow metric type{ens_type}")

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

def read_partial(in_path:str):
    if(type(in_path)==PartialResults):
        return in_path
    raw=np.load(in_path)
    y_partial,y_true=raw['name1'],raw['name2']
    return PartialResults(y_partial=y_partial,
                          y_true=y_true)

def read_result_group(in_path:str):
    results= [ read_result(path_i) 
                 for path_i in utils.top_files(in_path)]
    return ResultGroup(results)

def read_partial_group(in_path:str):
    results= [ read_partial(path_i) 
                 for path_i in utils.top_files(in_path)]
    return PartialGroup(results)

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

if __name__ == '__main__':
    data=read_csv("../uci/lymphography")
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        print(x_i.shape)