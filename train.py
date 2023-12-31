import utils
utils.silence_warnings()
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import base,data,deep,hyper

def train_models(data_path,model_path,multi=True):
    alg_params=base.AlgParams(hyper_type='eff')
    @utils.log_time(main_path=0)
    def helper(in_path,model_path):
        X,y=data.get_dataset(in_path)
        params=data.get_dataset_params(X,y)
        split_i= base.single_split(X=X, 
                                    y=y)
        hyper_dict=hyper.bayes_optim(alg_params=alg_params,
                                 split=split_i,
                                 params=params,
                                 verbose=0)
        exp_i=make_exp(alg_params=alg_params,
                    split_i=split_i,
                    params=params,
                    hyper_dict=hyper_dict)
        exp_i.save(model_path)
    if(multi):
        helper=utils.dir_fun(helper)
    helper(data_path,model_path)
#    model_path=model_path.split('.')[0]

def make_exp(alg_params,split_i,params,hyper_dict):
    if(alg_params.optim_alpha()):
        alpha_i,exp_i=hyper.find_alpha(alg_params=alg_params,
                                       split=split_i,
                                       params=params,
                                       hyper_dict=hyper_dict)
        exp_i.hyper_params['alpha']=alpha_i
    else:
        exp_i=base.Experiment(split=split_i,
                             params=params,
                             hyper_params=hyper_dict,
                             model=None)
        exp_i.train(verbose=0,
                    callbacks=alg_params.stop_early,
                    alpha=0.5)
    return exp_i

def raw_rf(split_i,n_select=5):    
    clf=get_clf("RF")
    x_train,y_train=split_i.get_train()
    clf.fit(x_train,y_train)
    x_valid,y_valid=self.get_valid()
    y_pred=clf.predict(x_valid)
    cf=confusion_matrix(y_train,y_pred)
    np.fill_diagonal(cf,0)
    return np.armax(np.sum(cf ,axis=0))

if __name__ == '__main__':
    utils.start_log('log.info')
    train_models(data_path='redu',
                 model_path='../OML_select',
                 multi=True)