import numpy as np
import argparse
import utils

def all_exp(in_path,stats=False):
#    for path_i in utils.top_files(in_path):
#        name_i=path_i.split('/')[-1]
    metric=utils.get_metric('acc')
    @utils.dir_fun
    def helper(in_path,out_path):
        result_i=np.load(in_path)
        acc_i=metric(result_i['true'],result_i['pred'])
        print(f'{in_path}:{acc_i:.4f}')
#        result_i=get_stats(in_path,#path_i,
#        	               verbose=0,
#                           stats=stats)
#        if(stats):
#            mean_i,std_i=result_i
#            print(f'{name_i},{mean_i:.4f},{std_i:.4f}')
#        else:
#            result_i.sort()
#            print(name_i+','.join([f'{acc_j:.4}' 
#                        for acc_j in result_i][-2:]))
    helper(in_path,'out')   

def get_stats(in_path,verbose=1,stats=False):
    metric=utils.get_metric('acc')
    acc=[]
    for path_i in utils.top_files(in_path):
        result_i=np.load(path_i)
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
if __name__ == '__main__':
#    all_exp('../OML_reduce/pred',False)
    all_exp('../OML_reduce/pred_optim',False)
