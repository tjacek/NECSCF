import numpy as np 
import keras
import tensorflow as tf
from sklearn.neighbors import BallTree
import deep

class PurityLoss(object):
    def __init__(self,multi=True):
        self.multi=multi
        self.hist=None

    def init(self,data):
        self.hist=purity_hist(data)

    def __call__(self,specific,class_dict):
        n_cats=len(class_dict)
        if(self.multi):
            class_weights=np.zeros(n_cats,dtype=np.float32)
            if(specific is None):
                for i in range(n_cats):
                    class_weights[i]=class_dict[i]
            else:
                purity_s=self.hist[specific,:]
                for i in range(n_cats):
                    if(i==specific):
                        class_weights[i]=2*class_dict[i]
                    else:
                        class_weights[i]=(1.0-purity_s[i])*class_dict[i]

            return deep.keras_loss(class_weights)
        else:
            class_dict=class_dict.copy()
            if(specific is None):
                return class_dict
            for i in range(n_cats):
                if(i!=specific):
                    purity_s=self.hist[specific,:]
                    class_dict[i]=(1.0-purity_s[i])*class_dict[i]
                else:
                    class_dict[i]=2*class_dict[i]
            return class_dict
    
    def __str__(self):
        name="purity_ens"
        if(not self.multi):
            name=f"separ_{name}"
        return name

def get_callback(callback_type):
    if(callback_type=="min"):
        return MinAccEarlyStopping
    if(callback_type=="all"):
        return ImprovEarlyStopping
    if(callback_type=="total"):
        return TotalEarlyStopping
    raise Exception(f"Unknow callback type{callback_type}")

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

class MinAccEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=15,
                       verbose=1):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.verbose=verbose

    def init(self,n_clfs):
        pass 

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        acc=[]
        for key_i in logs.keys():
            if("accuracy" in key_i):
                acc.append(logs[key_i])
        min_acc=np.amin(acc)
        if(min_acc>self.best):
            if(self.verbose):
                print(f"min_acc{min_acc},{self.wait}")
            self.best=min_acc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait+=1
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)


class ImprovEarlyStopping(keras.callbacks.Callback):
    def __init__(self,#n_clfs, 
                      patience=15,
                      eps=0.0001,
                      good_acc=0.95,
                      verbose=0):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.eps=eps
        self.good_acc=good_acc
        self.verbose=verbose
        
    def init(self,n_clfs):
        self.best=np.zeros(n_clfs,dtype=float)
        self.wait=np.zeros(n_clfs,dtype=int)
    
    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        for key_i in logs.keys():
            if("accuracy" in key_i):
                i=utils.extract_number(key_i)
                if(self.best[i]>self.good_acc):
                    self.wait[i]=self.patience+1
                    continue
                current_i=logs[key_i]
                diff_i= current_i-self.best[i]
                if(diff_i>= self.eps):
                    self.best[i]=current_i
                    self.wait[i]=0
                else:
                    self.wait[i]+=1
        if(self.verbose):
            print(f"epoch:{epoch}")
            print(self.best)
            print(self.wait)
        min_wait=np.amin(self.wait)
        if(min_wait>self.patience):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
        else:
            self.best_weights = self.model.get_weights()


class TotalEarlyStopping(keras.callbacks.Callback):
    def __init__(self,#n_clfs, 
                      patience=15,
                      eps=0.0001,
                      good_acc=0.95,
                      verbose=0):
        super().__init__()
        self.patience = patience
        self.best_weights = None
        self.eps=eps
        self.good_acc=good_acc
        self.verbose=verbose
    
    def init(self,n_clfs):
        self.best=0
        self.wait=0

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        total_acc=0.0
        for key_i in logs.keys():
            if("accuracy" in key_i):
                total_acc+=logs[key_i]
        diff= total_acc-self.best
        if(diff>= self.eps):
            self.best=total_acc
            self.wait=0
        else:
            self.wait+=1
        if(self.verbose):
            print(f"epoch:{epoch}")
            print(self.best)
            print(self.wait)
        if(self.wait>self.patience):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
        else:
            self.best_weights = self.model.get_weights()

def purity_hist(in_path,k=10):
    data_i=dataset.read_csv(in_path)
    tree=BallTree(data_i.X)
    indces= tree.query(data_i.X,
                           k=k+1,
                          return_distance=False)
    n_cats=data_i.n_cats()
    hist=np.zeros((n_cats,n_cats))
    sizes=np.zeros(n_cats)
    for i,ind_i in enumerate(indces):
        point_i=int(data_i.y[i])
        sizes[point_i]+=1
        for ind_j in ind_i[1:]:
            near_j=int(data_i.y[ind_j])
            hist[point_i][near_j]+=1
    hist/=k
    for i,size_i in enumerate(sizes):
        hist[i,:]/=size_i
    return hist