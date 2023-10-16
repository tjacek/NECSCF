import os,warnings

import logging,time
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from functools import wraps
import json

def silence_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tf.get_logger().setLevel('WARNING')#'ERROR')
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

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
    
def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

def by_col(df_i,name='clf'):
    return { clf_i:df_i[df_i[name]==clf_i]
              for clf_i in  df_i[name].unique()}

def log_time(task='TRAIN',main_path='in_path'):
    def helper(fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
#            raise Exception(kwargs)
            logger=logging.getLogger(__name__)
            if(type(main_path)==int):
                name_i=args[main_path].split('/')[-1]
            else:
                name_i=kwargs[main_path].split('/')[-1]
            start=time.time()
            result=fun(*args,**kwargs)
            diff=(time.time()-start)
            logger.info(f'{task}-{name_i}-{diff:.4f}s')
            return result
        return decor_fun
    return helper

def dir_fun(fun):
    @wraps(fun)
    def decor_fun(*args, **kwargs):
        in_path,out_path= args[0],args[1]
        make_dir(out_path)
        result_dict={}
        for path_i in top_files(in_path):
            name_i=path_i.split('/')[-1]
            out_i=f'{out_path}/{name_i}'
            new_args=list(args)
            new_args[0]=path_i
            new_args[1]=out_i
            result_dict[name_i]=fun(*new_args,**kwargs)
        return result_dict
    return decor_fun

class FileFilter:
    def __call__(self, log):
        if log.levelno < logging.WARNING:
            return 1
        else:
            return 0

def start_log(log_path):
    logging.basicConfig(level=logging.INFO,
                        filemode='a', 
                        format='%(process)d-%(levelname)s-%(message)s')
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(log_path))
    logger.info('test')
    logging.warning('warn')
