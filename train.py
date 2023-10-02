import utils
utils.silence_warnings()
import numpy as np
import json
import base,data,deep,hyper

@utils.dir_fun
def find_hyper(data_path,model_path,alpha_optim=True):
    X,y=data.get_dataset(data_path)
    params=data.get_dataset_params(X,y)

    split_i= base.single_split(X=X, 
                               y=y)
#    split_i.is_valid()
    hyper_dict=hyper.bayes_optim(split=split_i,
                                 params=params,
                                 n_iter=5,
                                 verbose=0)
    if(alpha_optim):
        alpha_i,exp_i=hyper.find_alpha(split=split_i,
                                   params=params,
                                   hyper_dict=hyper_dict)
        exp_i.hyper_params['alpha']=alpha_i
    else:
        exp_i=base.Experiment(split=split_i,
                             params=params,
                             hyper_params=hyper_dict,
                             model=None)
        stop_early = deep.get_early_stop()
        exp_i.train(verbose=0,
                    callbacks=stop_early,
                    alpha=0.5)
#    model_path=model_path.split('.')[0]
    exp_i.save(model_path)

#def all_train(in_path,out_path,n_iters=2):
#    names={'arrhythmia':-1}#,'mfeat-factors':-1,'vehicle':-1,
#           'cnae-9':-1,'car':-1,'segment':-1,'fabert':0}
#    for name_i,target_i in names.items():
#        train_exp(in_path=f'{in_path}/{name_i}.arff',
#                  out_path=f'{out_path}/{name_i}',
#                  n_iters=n_iters,
#                  target=-1 )

#@utils.log_time(task='TRAIN')
#def train_exp(in_path,out_path,n_iters=2,hyper=None,target=-1 ):
#    df=data.from_arff(in_path)
#    X,y=data.prepare_data(df,target=target)
#    params=data.get_dataset_params(X,y)
#    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
#                                                  patience=5)
#    utils.make_dir(out_path)
#    for i,split_i in enumerate(gen_split(X,y,n_iters)):
#        exp_i=Experiment(split=split_i,
#        	             params=params)	
#        exp_i.find_hyper()
#        exp_i.train(verbose=1,
#                    callbacks=stop_early)
#        exp_i.save(f'{out_path}/{i}')

if __name__ == '__main__':
#    name='arrhythmia'#cnae-9'
#    in_path=f'raw/{name}.arff'
    find_hyper('redu','../OML_reduce')
#    utils.start_log('log.info')
#    all_train(in_path='raw', #'../OML/models',
#              out_path='../OML/_models',
#              n_iters=2)
#    train_exp(in_path=in_path,
#    	      out_path=f'../OML/models/{name}',
#              n_iters=10,
#              target=-1)