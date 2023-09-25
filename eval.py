import numpy as np
import utils

def all_exp(in_path):
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        mean_i,std_i=get_stats(path_i,
        	                   verbose=0)
        print(f'{name_i},{mean_i:.2f},{std_i:.2f}')
        	
def get_stats(in_path,verbose=1):
    metric=utils.get_metric('acc')
    acc=[]
    for path_i in utils.top_files(in_path):
        result_i=np.load(path_i)
#        print(result_i['true'])
#        print(result_i['pred'])
        acc_i=metric(result_i['true'],result_i['pred'])
        acc.append(acc_i)
    mean_i,std_i=np.mean(acc),np.std(acc)
    if(verbose):
        print(f'Mean:{mean_i}')
        print(f'Std:{std_i}')
    return mean_i,std_i

#name='cnae-9'
#get_stats(f'../OML/pred/{name}')
all_exp('../OML/pred')