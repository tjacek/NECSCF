import utils
utils.silence_warnings()
import numpy as np
import json
import base,data,deep,hyper

@utils.dir_fun
def find_hyper(data_path,model_path,alg_params):
    X,y=data.get_dataset(data_path)
    params=data.get_dataset_params(X,y)
    split_i= base.single_split(X=X, 
                               y=y)
    hyper_dict=hyper.bayes_optim(alg_params=alg_params,
                                 split=split_i,
                                 params=params,
                                 n_iter=5,
                                 verbose=0)
    exp_i=train_exp(alg_params=alg_params,
                    split_i=split_i,
                    params=params,
                    hyper_dict=hyper_dict)

#    model_path=model_path.split('.')[0]
    exp_i.save(model_path)

def train_exp(alg_params,split_i,params,hyper_dict):
    if(alg_params.alpha_optim()):
        alpha_i,exp_i=hyper.find_alpha(split=split_i,
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

if __name__ == '__main__':

#    utils.start_log('log.info')
