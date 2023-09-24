import numpy as np
import utils

def show_stats(in_path):
    metric=utils.get_metric('acc')
    acc=[]
    for path_i in utils.top_files(in_path):
        result_i=np.load(path_i)
        acc_i=metric(result_i['true'],result_i['pred'])
        acc.append(acc_i)
    print(f'Mean:{np.mean(acc)}')
    print(f'Std:{np.std(acc)}')

in_path='pred'
show_stats(in_path)