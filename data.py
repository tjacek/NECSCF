import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing

def get_dataset(data_path):
    df=pd.read_csv(data_path,header=None) 
    return prepare_data(df)

def prepare_data(df,target=-1):
    X=df.iloc[:,:target]
    print(X)
    X=X.to_numpy()
    X=preprocessing.scale(X) # data preprocessing
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,
            'dims':X.shape[1],
            'batch':X.shape[0],
            'class_weights':dict(Counter(y))}

def from_arff(in_path:str):
    raw_rows,raw_cats = arff.loadarff(in_path)   
    rows=[ [float(r_j) for r_j in raw_i]
           for raw_i in raw_rows]    
#    X=np.array(dataset[0])
    cols=[col_j.split('\'')[0] for col_j in raw_cats] 
    df=pd.DataFrame(rows,columns=cols)
    print(df.head())
    
if __name__ == '__main__':
    in_path='raw/compas.arff'
    from_arff(in_path)