import numpy as np
import pandas as pd 
from collections import namedtuple
from collections import defaultdict

ExpResult = namedtuple("exp_result", "mean std")

def parse_exp(raw_i):
    mean,std=raw_i.split('±')
    return ExpResult(float(mean),float(std))

def txt_exp(exp_i):
    return f'{exp_i.mean}±{exp_i.std}'

def parse_compa(df):
    compa_dict={}
    for i,row_i in df.iterrows():
        dict_i=row_i.to_dict()
        name_i=dict_i['Datatset']
        compa_dict[name_i]= { name_j.strip():parse_exp(raw_j) 
                        for name_j,raw_j in dict_i.items()
                            if(name_j!='Datatset')}
    return compa_dict

def get_ne( ne_df):
    ne_result=ne_df['NECSCF'].tolist()
    ne_result= [ parse_exp(ne_i) for ne_i in ne_result]	
    names=[name_i.strip() 
            for name_i in ne_df['Dataset'].tolist()]
    return dict(zip(names,ne_result))

def find_best(dict_i):
    mean_dict={name_i:exp_i.mean for name_i,exp_i in dict_i.items()}
    names,values= list(zip(*mean_dict.items()))
    k=np.argmax(values)
    return names[k]

df=pd.read_csv('wozniak.csv')
ne_df=pd.read_csv('rf.csv')


compa_dict= parse_compa(df)

best_dict=get_ne(ne_df)

table_dict=defaultdict(lambda :[]) #{col_i:[]  for col_i in [data,alg,alg_exp,better,sig}
for name_i,exp_i in best_dict.items():
    comp_i= compa_dict[name_i]
    best_i=find_best(comp_i)
    best_exp=comp_i[best_i]
    table_dict['data'].append(name_i)
    table_dict['alg'].append(best_i)
    table_dict['alg_acc'].append(txt_exp(best_exp))
    table_dict['NECSCF_acc'].append(txt_exp(exp_i))
    table_dict['better'].append( exp_i.mean > best_exp.mean)
    table_dict['sig'].append( exp_i.mean > best_exp.mean - best_exp.std)
    table_dict['best'].append( exp_i.mean > best_exp.mean + best_exp.std)

result_df=pd.DataFrame.from_dict(table_dict)
print(result_df)
print(result_df[['data','alg','alg_acc','NECSCF_acc']].to_latex())