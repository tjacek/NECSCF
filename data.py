import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from collections import Counter
import utils

def get_dataset(data_path):
    postfix=data_path.split('.')[-1]
    if(postfix=='arff'):
        df=from_arff(data_path)
        return prepare_data(df)
    elif(postfix=='npz'):
        np_data=np.load(data_path)
        return np_data['X'],np_data['y']
    else:
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

def gini(params):
    class_sizes=list(params['class_weights'].values())
    class_sizes.sort()
    height,area = 0,0
    for value in class_sizes:
        height += value
        area += height - value / 2.
    fair_area = height * len(class_sizes) / 2.
    return (fair_area - area) / fair_area

def show_prop(in_path):
    @utils.dir_fun
    def helper(in_path,out_path):
        X,y=get_dataset(in_path)
        params=get_dataset_params(X,y)
        params['gini']=gini(params)
        return params
    result_dict=helper(in_path,'out')
    import pandas as pd
    raw_dict={ name_i:[]
         for name_i in ['dataset','n_cats','dims','batch','gini']}
    for name_i,dict_i in result_dict.items():
        raw_dict['dataset'].append(name_i)
        for name_j in ['n_cats','dims','batch','gini']:
            raw_dict[name_j].append(dict_i[name_j])
    import pandas as pd
    df=pd.DataFrame.from_dict(raw_dict)
    return df

def show_hyper(in_path):
    import json
    @utils.dir_fun
    def helper(in_path,out_path):
        with open(f'{in_path}/hyper.json', 'r') as f:        
            json_bytes = f.read()                      
            return json.loads(json_bytes)
    result_dict=helper(in_path,in_path)
    cols=list(result_dict.values())[0].keys()
    raw_dict={ name_i:[] for name_i in cols}
    raw_dict['dataset']=[]
    for name_i,dict_i in result_dict.items():
        raw_dict['dataset'].append(name_i)
        for name_j in cols:
            raw_dict[name_j].append(dict_i[name_j])
    import pandas as pd
    df=pd.DataFrame.from_dict(raw_dict)
    df=df[['dataset','units_0','units_1','batch','alpha']]
    return df

if __name__ == '__main__':
    df=show_hyper('../OML_reduce/models')
    print(df.to_latex())
#    df=show_prop('raw')
#    print(df[[ 'dataset','n_cats','batch', 'dims','gini' ]].to_latex())
