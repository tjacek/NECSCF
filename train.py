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

#def malloc_trim():
#    ctypes.CDLL('libc.so.6').malloc_trim(0) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci/cleveland")
    parser.add_argument("--out_path", type=str, default="exp_deep/cleveland")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    print(args)
    base_train(data_path=args.data,
               out_path=args.out_path,
               start=args.start,
               step=args.step)