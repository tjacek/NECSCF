import numpy as np
import pandas as pd
from scipy import stats
import utils

SEP='|'

class MetricDict(dict):
    def __init__(self, arg=[]):
        super(MetricDict, self).__init__(arg)
        self.cols=['dataset','cs','clf','variant']
    
    def to_df(self,taboo=None):
        lines=[]
        stats=[np.mean,np.std]
        for id_i,metric_i in self.items():
            if(id_i.is_valid(taboo)):
                line_i=id_i.split('|')
                line_i+=[round(stat_j(metric_i),4) 
                        for stat_j in stats]
                line_i.append(id_i)
                lines.append(line_i)
        cols=self.cols+['mean','std','id']
        df=pd.DataFrame(lines,columns=cols)
        return df

    def pvalue(self,id_x,id_y):
        if( (not id_x in self) or (not id_y in self)):
            raise Exception(f'{id_x}#{id_y}')
        r=stats.ttest_ind(self[id_x], 
                self[id_y], equal_var=False)
        p_value=round(r[1],4)
        return p_value

class ExpID(str):
    def __getitem__(self, key):
        return self.split(SEP)[key]
    
    def is_valid(self,taboo):
        if(not taboo):
            return True 
        raw=self.split(SEP)
        for i,taboo_i in enumerate( taboo):
            for taboo_j	in taboo_i:
                if(raw[i]==taboo_j):
                    return False	
        return True

def read_acc(pred_path,metric_type='acc'):
    metric_i=utils.get_metric(metric_type)
    acc_dict=MetricDict()
    for path_i in utils.top_files(pred_path):
        data_i=path_i.split('/')[-1]
        for path_j in utils.top_files(path_i):
            id_i=path_j.split('/')[-1]
            if('base' in id_i):
                exp_id=[data_i,'-',id_i.split('-')[-1],'-']
            else:
                exp_id=[data_i]+id_i.split('-')
            exp_id=ExpID('|'.join(exp_id)) 	
            all_pred=utils.read_pred(path_j)
            acc_ij=[ metric_i(test_i,pred_i) 
                       for test_i,pred_i in all_pred]
            acc_dict[exp_id]=acc_ij
    return acc_dict

def pvalues(df,acc_dict):
    cols=acc_dict.cols+['pvalue','sig','diff']
    for data_i,data_df in utils.by_col(df,'dataset').items():
        for clf_i,clf_df in utils.by_col(data_df,'clf').items():
            base,variant=get_ids(clf_df)
            if(base):
                lines=[]
                for variant_k in variant:
                    pvalue_k=acc_dict.pvalue(base[0],
                    	                     variant_k)
                    diff_k=check_diff(clf_df,
                    	              base[0],
                    	              variant_k)
                    line_k=variant_k.split('|')
                    line_k+=[ pvalue_k,
                              int(pvalue_k<0.05),
                              diff_k]
                    lines.append(line_k)
                df_i=pd.DataFrame(lines,
                	         columns=cols)
                yield df_i

def get_ids(df_i):
    variant= df_i[df_i['variant']!='-']['id'].tolist()
    base= df_i[df_i['variant']=='-']['id'].tolist()
    return base,variant

def check_diff(clf_df,id_x,id_y):
    x= clf_df[clf_df['id']==id_x]['mean'] 
    x=x.tolist()[0]
    y= clf_df[clf_df['id']==id_y]['mean'] 
    y=y.tolist()[0]
    return round(y-x,4)

def pvalue_stats(pvalue_df):
    counter=[0,0,0,0]
    for p_df in pvalue_df:
        print(p_df)	
        for i,row_i in p_df.iterrows():
            if(row_i[-2]>0 and row_i[-1]>0):
                counter[0]+=1
            if(row_i[-2]==0 and row_i[-1]>0):
                counter[1]+=1
            if(row_i[-2]>0 and row_i[-1]<0):
                counter[2]+=1
            if(row_i[-2]==0 and row_i[-1]<0):
                counter[3]+=1		
    print(counter)

def summary(taboo,acc_dict):
    df=acc_dict.to_df(taboo)
    cols=['dataset','mean','std']
    df_rf=df[df['variant']=='-'][cols] 
    df_ne=df[df['variant']=='necscf'][cols]
    pvalue_df= pd.concat(pvalues(df,acc_dict))
    pvalue_df=pvalue_df[['dataset','pvalue','sig']]
    df=pd.merge(df_rf, df_ne, on=["dataset"])
    df=pd.merge(df, pvalue_df, on=["dataset"])
    print(df.to_latex())

if __name__ == '__main__':
    dir_path='../pred'
    acc_dict= read_acc(dir_path,metric_type='balanced')

    taboo=[[],
       ['binary',
        'multi'],

#        'weighted']
       ['SVC','LR'],
       ['cs','inliner']] #'necscf']]#]]
    summary(taboo,acc_dict)