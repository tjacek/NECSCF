import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

class UnaggrSplit(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(self.Split(train_index,test_index))
        return splits

    class Split(object):
        def __init__(self,train_index,test_index):
            self.train_index=train_index
            self.test_index=test_index

        
        def eval(self,data,clf):
            return data.eval(train_index=self.train_index,
                             test_index=self.test_index,
                             clf=clf,
                             as_result=True)
       
        def fit_clf(self,data,clf):
           return data.fit_clf(self.train_index,clf)

        def pred(self,data,clf):
            return data.pred(self.test_index,
                             as_result=True)

class AggrSplit(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                       n_splits=self.n_splits, 
                                       random_state=0)
        splits=[]
        for t,(train_index,test_index) in enumerate(rskf.split(data.X,data.y)):
            if((t % self.n_splits)==0):
            	splits.append([])
            splits[-1].append((train_index,test_index))
        splits=[self.Split(indexes) for indexes in splits]
        return splits

    class Split(object):
        def __init__(self,indexes):
            self.indexes=indexes

        def eval(self,data,clf):
            all_clf=self.fit_clf(data,clf)
            return self.pred(data,all_clf)

        def fit_clf(self,data,clf):
            return [ data.fit_clf(train_t) 
                        for train_t,_ in self.indexes]

        def pred(self,data,clf):
            all_pred,all_test=[],[]
            for i,(_,test_t) in enumerate(self.indexes):
                pred_t,test_t=data.pred(test_index=test_t,
                                        clf=clf[i],
                                        as_result=False)
                all_pred.append(pred_t)
                all_test.append(test_t)
            all_pred=np.concatenate(all_pred)
            all_test=np.concatenate(all_test)
            return dataset.Result(all_pred,all_test)

#        def eval(self,data,clf):
#            all_pred,all_test=[],[]
#            for train_t,test_t in self.indexes:
#                pred_t,test_t=data.eval(train_index=train_t,
#                                        test_index=test_t,
#                                        clf=clf,
#                                        as_result=False)
#                all_pred.append(pred_t)
#                all_test.append(test_t)
#            all_pred=np.concatenate(all_pred)
#            all_test=np.concatenate(all_test)
#            return dataset.Result(all_pred,all_test)

def get_protocol(prot_type):
    if(prot_type=="aggr"):
        return AggrSplit
    if(prot_type=="unaggr"):
        return UnaggrSplit
    raise Exception(f"No protocol{prot_type}")