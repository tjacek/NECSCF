import pandas as pd 
from collections import namedtuple

ExpResult = namedtuple("exp_result", "mean std")

def parse_exp(raw_i):
    print(raw_i)	
    mean,std=raw_i.split('±')
    return ExpResult(float(mean),float(std))

def parse_compa(df):
    compa_dict={}
    for i,row_i in df.iterrows():
        dict_i=row_i.to_dict()
        name_i=dict_i['Datatset']
        compa_dict[name_i]= { name_j.strip():parse_exp(raw_j) 
                        for name_j,raw_j in dict_i.items()
                            if(name_j!='Datatset')}
    return compa_dict

def to_exp( raw):
    return [ parse_exp(raw_i) for raw_i in raw]	

df=pd.read_csv('wozniak.csv')
ne_df=pd.read_csv('rf.csv')

ne_result=ne_df['NECSCF'].tolist()

compa_dict= parse_compa(df)

#print(to_exp( ne_result))