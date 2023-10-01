import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import data,utils,train

def compare(in_path,select):
    acuracy=utils.get_metric('acc')
    for path_i in utils.top_files(in_path):
        X,y=data.get_dataset(path_i)
        split_i= list(train.gen_split(X,y,n_iters=1))[0]
        y_true,y_pred= split_i.eval('RF')
        acc_i=acuracy(y_true,y_pred)
        print(f'{acc_i:.4f},{X.shape[1]}')
        X=select(X,y)
        split_i.X=X
        y_true,y_pred= split_i.eval('RF')
        acc_i=acuracy(y_true,y_pred)
        print(f'{acc_i:.4f},{X.shape[1]}')      

def transform(in_path,out_path,select):
    utils.make_dir(out_path)	
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        X,y=data.get_dataset(path_i)
        X=select(X,y)
        np.savez(file=f'{out_path}/{name_i}',
                 X=X,
                 y=y)

def kbest(X,y,k=100):
    if(X.shape[1]<k):
        return X	
    return SelectKBest(f_classif, k=k).fit_transform(X, y)

#compare('raw',kbest)
transform('raw','redu',kbest)