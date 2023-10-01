import numpy as np
import utils

def all_exp(in_path,stats=True):
    for path_i in utils.top_files(in_path):
        name_i=path_i.split('/')[-1]
        result_i=get_stats(path_i,
        	               verbose=0,
                           stats=stats)
        if(stats):
            mean_i,std_i=result_i
            print(f'{name_i},{mean_i:.4f},{std_i:.4f}')
        else:
            result_i.sort()
            print(name_i+','.join([f'{acc_j:.4}' 
                        for acc_j in result_i][-2:]))

def get_stats(in_path,verbose=1,stats=False):
    metric=utils.get_metric('acc')
    acc=[]
    for path_i in utils.top_files(in_path):
        result_i=np.load(path_i)
#        print(result_i['true'])
#        print(result_i['pred'])
        acc_i=metric(result_i['true'],result_i['pred'])
        acc.append(acc_i)
    if(not stats):
        return acc
    mean_i,std_i=np.mean(acc),np.std(acc)
    if(verbose):
        print(f'Mean:{mean_i}')
        print(f'Std:{std_i}')
    return mean_i,np.max(acc)#std_i

#name='cnae-9'

all_exp('../OML/pred',False)
all_exp('../OML/optim_pred',False)
