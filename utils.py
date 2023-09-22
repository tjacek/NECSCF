import os
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import json

def get_metric(metric_i):
    if(metric_i=='acc'):
        return accuracy_score
    elif(metric_i=='balanced'):
        return balanced_accuracy_score

def top_files(path):
    if(type(path)==str):
        paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

def by_col(df_i,name='clf'):
    return { clf_i:df_i[df_i[name]==clf_i]
              for clf_i in  df_i[name].unique()}