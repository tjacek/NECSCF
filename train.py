import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import gc#,ctypes
import argparse
import base,ens,utils

#@utils.DirFun({"data_path":0,"out_path":1},input_arg='data_path')
def train_models(data_path,
                 out_path,
                 n_splits=10,
                 n_repeats=10):
    data_split=base.get_splits(data_path=data_path,
                           n_splits=n_splits,
                           n_repeats=n_repeats)
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data_split.data)
    utils.make_dir(out_path)
    for i,split_i in tqdm(enumerate(data_split.splits)):
        out_i=f"{out_path}/{i}"
        clf_i=clf_factory()
        clf_i=split_i.fit_clf(data_split.data,clf_i)
        utils.make_dir(out_i)
        split_i.save(f"{out_i}/split")
        clf_i.save(f"{out_i}/model.h5")
        del clf_i
        gc.collect()
#        malloc_trim()

#def malloc_trim():
#    ctypes.CDLL('libc.so.6').malloc_trim(0) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci/cleveland")
    parser.add_argument("--model", type=str, default="../full_exp/cleveland")
    args = parser.parse_args()
    print(args)
    train_models(data_path=args.data,
                 out_path=args.model)