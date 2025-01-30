import tensorflow as tf
from keras import Input, Model
import base,dataset,deep,ens

def show_loss(data_path,split_path):
    data_split=base.read_data_split(data_path=data_path,
		                            split_path=split_path)
    loss_dict={'base':base_loss,'custom':custom_loss}
    for i,data_i in data_split.selection_iter(train=True):
        for type_j,loss_j in loss_dict.items():

            model_i,params=get_model(data_i)
            loss_j(model_i,params)
            y=tf.one_hot(data_i.y,depth=params['n_cats'])
            history=model_i.fit(x=data_i.X,
        	                y=y)
            print(history.history.keys())

def base_loss(model,params):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def custom_loss(model,params):
    loss=deep.weighted_loss(specific=None,
                       class_dict=params['class_weights'])
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def get_model(data):
    params={'dims': (data.dim(),),
            'n_cats':data.n_cats(),
            'n_epochs':100,
            'class_weights':dataset.get_class_weights(data.y)}
    hyper_params=ens.default_hyperparams()
    input_layer = Input(shape=(params['dims']))
    nn=deep.nn_builder(params=params,
                    hyper_params=hyper_params,
                    input_layer=input_layer,
                    i=0,
                    n_cats=params['n_cats'])
    model= Model(inputs=input_layer, 
                outputs=nn)
    return model,params

show_loss(data_path="../uci/wall-following",
	      split_path="single_exp/wall-following/splits")