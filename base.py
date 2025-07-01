import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from tqdm import tqdm
import dataset,utils

class DataSplits(object):
    def __init__(self,data,splits):
        self.data=data
        self.splits=splits
        
    def __call__(self,clf_factory):
        clf_factory.init(self.data)
        results,history=[],[]
        for split_k in tqdm(self.splits):
            clf_k=clf_factory()
            result_k,history_k=split_k.eval(self.data,clf_k)
            results.append(result_k)
            history.append(utils.history_to_dict(history_k))
        results=dataset.ResultGroup(results)
        return results,history

    def get_clfs(self,clf_factory):
        clf_factory.init(self.data)
        for split_k in self.splits:
            clf_k=clf_factory()
            yield split_k.fit_clf(self.data,clf_k)

    def pred(self,clfs):
        results=[]
        for split_k,clf_k in zip(self.splits,clfs):
            results.append(split_k.pred(self.data,clf_k))
        return dataset.ResultGroup(results)

    def iter(self,fun,clf_factory):
        clf_factory.init(self.data)
        for split_k in tqdm(self.splits):
            clf_k=clf_factory()
            yield fun(split_k,clf_k)        

    def selection_iter(self,train=False):
        for i,split_i in enumerate(self.splits):
            if(train):
                index=split_i.train_index
            else:
                index=split_i.test_index
            yield  i,self.data.selection(index)

    def basic_eval(self,clf_type):
        results=[]
        for split_j in self.splits:
            clf_j=get_clf(clf_type)
            result_j,_=split_j.eval(self.data,clf_j)
            results.append(result_j)
        return dataset.ResultGroup(results)

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
                             clf=clf,
                             as_result=True)

        def save(self,out_path):
            return np.savez(out_path,self.train_index,self.test_index)

        def __str__(self):
            train_size=self.train_index.shape[0]
            test_size=self.test_index.shape[0]
            return f"train:{train_size},test:{test_size}"

def get_protocol(prot_type):
    if(prot_type=="aggr"):
        return AggrSplit
    if(prot_type=="unaggr"):
        return UnaggrSplit
    raise Exception(f"No protocol{prot_type}")

def get_clf(clf_type):
    if(clf_type=="RF"): 
        return RandomForestClassifier(class_weight="balanced")#_subsample")
    if(clf_type=="LR"):
        return LogisticRegression(solver='liblinear')
    if(clf_type=="SVM"):
        return svm.SVC(kernel='rbf')
    if(clf_type=="GRAD"):
        return GradientBoostingClassifier()

    raise Exception(f"Unknow clf type:{clf_type}")

def get_splits(data_path,
               n_splits=10,
               n_repeats=1,
               split_type="unaggr"):
    data=dataset.read_csv(data_path)
    protocol=get_protocol(split_type)(n_splits,n_repeats)
    return DataSplits(data=data,
                      splits=protocol.get_split(data))

def read_data_split(data_path,split_path):
    data=dataset.read_csv(data_path)
    splits=[ read_split(path_i) for path_i in utils.top_files(split_path)]
    return DataSplits(data=data,
                      splits=splits)    
 
def read_split(in_path):
    raw_split=np.load(in_path)
    return UnaggrSplit.Split(train_index=raw_split["arr_0"],
                             test_index=raw_split["arr_1"])

def get_paths(out_path,ens_type,dirs):
    ens_path=f"{out_path}/{ens_type}"
    path_dir={dir_i:f"{ens_path}/{dir_i}" 
                    for dir_i in dirs}
    path_dir['ens']=ens_path
    path_dir['splits']=f"{out_path}/splits"
    return path_dir