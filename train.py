import utils
utils.silence_warnings()
import numpy as np
#import json
#from sklearn.metrics import confusion_matrix
#import base,data,deep,hyper
import base,ens

def train_models(data_path,
                 out_path,
                 n_splits=10,
                 n_repeats=1):
    data_split=base.get_splits(data_path=data_path,
                           n_splits=n_splits,
                           n_repeats=n_repeats)
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data_split.data)
    utils.make_dir(out_path)
    for i,split_i in enumerate(data_split.splits):
        clf_i=clf_factory()
        split_i.fit_clf(data_split.data,clf_i)
        out_i=f"{out_path}/{i}"
        utils.make_dir(out_i)
        split_i.save(f"{out_i}/split")
        clf_i.save(f"{out_i}/model.h5")

#    alg_params=base.AlgParams(hyper_type='eff')
#    @utils.log_time(main_path=0)
#    def helper(in_path,model_path):
#        X,y=data.get_dataset(in_path)
#        params=data.get_dataset_params(X,y)
#        split_i= base.single_split(X=X, 
#                                    y=y)
#        hyper_dict=hyper.bayes_optim(alg_params=alg_params,
#                                 split=split_i,
#                                 params=params,
#                                 verbose=0)
#        exp_i=make_exp(alg_params=alg_params,
#                    split_i=split_i,
#                    params=params,
#                    hyper_dict=hyper_dict)
#        exp_i.save(model_path)
#    if(multi):
#        helper=utils.dir_fun(helper)
#    helper(data_path,model_path)

#def raw_rf(split_i,n_select=5):    
#    clf=get_clf("RF")
#    x_train,y_train=split_i.get_train()
#    clf.fit(x_train,y_train)
#    x_valid,y_valid=self.get_valid()
#    y_pred=clf.predict(x_valid)
#    cf=confusion_matrix(y_train,y_pred)
#    np.fill_diagonal(cf,0)
#    return np.armax(np.sum(cf ,axis=0))

if __name__ == '__main__':
    train_models(data_path="../uci/wine-quality-red",
                 out_path="exp")