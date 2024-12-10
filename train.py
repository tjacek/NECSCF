import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import base,ens,utils

@utils.DirFun({"data_path":0,"out_path":1},input_arg='data_path')
def train_models(data_path,
                 out_path,
                 n_splits=2,
                 n_repeats=2):
    data_split=base.get_splits(data_path=data_path,
                           n_splits=n_splits,
                           n_repeats=n_repeats)
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data_split.data)
    utils.make_dir(out_path)
    for i,split_i in tqdm(enumerate(data_split.splits)):
        clf_i=clf_factory()
        split_i.fit_clf(data_split.data,clf_i)
        out_i=f"{out_path}/{i}"
        utils.make_dir(out_i)
        split_i.save(f"{out_i}/split")
        clf_i.save(f"{out_i}/model.h5")

if __name__ == '__main__':
    train_models(data_path="../uci",
                 out_path="test_exp")