import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from collections import Counter
def get_dataset(data_path):
    df=pd.read_csv(data_path,header=None) 
    return prepare_data(df)

def prepare_data(df,target=-1):
    to_numeric(df)
    X=df.to_numpy()
    X=np.delete(X,[target], axis=1)
    X=np.nan_to_num(X)
#    X=preprocessing.scale(X) # data preprocessing
#    print(f'X:{np.mean(X)}')
    X= preprocessing.RobustScaler().fit_transform(X)
    y=df.iloc[:,target]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)

def to_numeric(df):
    for col_i,type_i in zip(df.columns,df.dtypes):
        if(type_i=='object'):
            values={value_i:i 
                 for i,value_i in enumerate(df[col_i].unique())}
            df[col_i]=df[col_i].apply(lambda x: values[x])
    return df

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,
            'dims':X.shape[1],
            'batch':X.shape[0],
            'class_weights':dict(Counter(y))}

def from_arff(in_path:str):
    raw_rows,raw_cats = arff.loadarff(in_path)   
    rows=[[convert(r_j) for r_j in raw_i]
            for raw_i in raw_rows]    
    cols=[col_j.split('\'')[0] for col_j in raw_cats] 
    df=pd.DataFrame(rows,columns=cols)
    return df

def convert(raw):
    try:
        return float(raw)
    except:
         return raw

if __name__ == '__main__':
    in_path='raw/compas.arff'
    from_arff(in_path)