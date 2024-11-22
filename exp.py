import dataset,deep


data=dataset.read_csv("../uci/cmc")
params={'dims': (data.dim(),),
        'n_cats':data.n_cats(),
        'class_weights':dataset.get_class_weights(data.y) }
model=deep.ensemble_builder(params,
	                        hyper_params=None,
	                        alpha=0.5)
model.summary()