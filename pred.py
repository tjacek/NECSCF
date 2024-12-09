import utils
utils.silence_warnings()
import numpy as np
import tensorflow as tf
import base,dataset,ens

def pred(data_path,model_path):
    data=dataset.read_csv(data_path)
    ens_factory=ens.ClassEnsFactory()
    ens_factory.init(data)
    acc=[]
    for split_i,clf_i in model_iter(model_path,ens_factory):
        result_i=split_i.pred(data,clf_i)
        acc.append(result_i.get_acc())
    print(np.mean(acc))

def model_iter(model_path,ens_factory):
    for path_i in utils.top_files(model_path):
        raw_split=np.load(f"{path_i}/split.npz")
        split_i=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        model_i=tf.keras.models.load_model(f"{path_i}/model.h5")
        clf_i=ens_factory()
        clf_i.model=model_i
        yield split_i,clf_i

#def all_pred(in_path,out_path):
#    metric=utils.get_metric('acc')
#    @utils.dir_fun
#    def helper(in_path,out_path):
#        exp_i=base.read_exp(in_path)
#        y_true,y_pred=nescf_eval(exp_i)
#        print(f'{out_path} Acc:{metric(y_true,y_pred):.4f}')
#        np.savez(file=out_path,
#                 true=y_true,
#                 pred=y_pred)
#    helper(in_path,out_path)

#def nescf_eval(exp):
#    extractor=exp.make_extractor() 
#    train,test=exp.split.extract(extractor=extractor,
#                                 use_valid=False)
#    (x_train,y_train),(x_test,y_test)=train,test
#    y_pred=base.simple_necscf(x_train=x_train,
#                              y_train=y_train,
#                              x_test=x_test,
#                              clf_type="RF")
#    return y_test,y_pred

#def common_exp(exp):
#    x_train,y_train=exp.split.get_train()
#    x_test,y_test=exp.split.get_test()
#    clf=make_clf(x_train,y_train)
#    y_pred= clf.predict(x_test)
#    return y_test,y_pred

#def nn_exp(exp):
#    x_test,y_test=exp.split.get_test()
#    votes=exp.model.predict(x_test)  
#    y_pred=base.count_votes(votes)
#    return y_test,y_pred

#def make_clf(split):#X,y):
#    X,y=split.get_train()
#    clf=ensemble.RandomForestClassifier(class_weight='balanced_subsample')
#    clf.fit(X,y)
#    return clf

if __name__ == '__main__':
    pred(data_path="../uci/wine-quality-red",
         model_path="exp")