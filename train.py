import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
#import gc#,ctypes
import argparse
import base,dataset,ens

def base_train(data_path:str,
               out_path:str,
               start=0,
               step=10):
    data=dataset.read_csv(data_path)
    split_path=f"{out_path}/splits"
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data)
    model_path=f"{out_path}/class_ens"
    utils.make_dir(model_path)
    for j in tqdm(range(step)):
        index=start+j
        raw_split=np.load(f"{split_path}/{index}.npz")
        split_j=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        clf_j=clf_factory()
        clf_j=split_j.fit_clf(data,clf_j)  
        clf_j.save(f"{model_path}/{index}.keras")
#def train_models(data_path,
#                 out_path,
#                 n_splits=10,
#                 n_repeats=10):
#    data_split=base.get_splits(data_path=data_path,
#                           n_splits=n_splits,
#                           n_repeats=n_repeats)
#    clf_factory=ens.ClassEnsFactory()
#    clf_factory.init(data_split.data)
#    utils.make_dir(out_path)
#    for i,split_i in tqdm(enumerate(data_split.splits)):
#        out_i=f"{out_path}/{i}"
#        clf_i=clf_factory()
#        clf_i=split_i.fit_clf(data_split.data,clf_i)
#        utils.make_dir(out_i)
#        split_i.save(f"{out_i}/split")
#        clf_i.save(f"{out_i}/model.h5")
#        del clf_i
#        gc.collect()

#def malloc_trim():
#    ctypes.CDLL('libc.so.6').malloc_trim(0) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci/cleveland")
    parser.add_argument("--out_path", type=str, default="exp_deep/cleveland")
    args = parser.parse_args()
    base_train(data_path=args.data,
               out_path=args.out_path)