import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
#import gc#,ctypes
import os.path
import argparse
import base,dataset,ens,utils

def base_train(data_path:str,
               out_path:str,
               start=0,
               step=10):
    data=dataset.read_csv(data_path)
    split_path=f"{out_path}/splits"
    clf_factory=ens.ClassEnsFactory()
    clf_factory.init(data)
    model_path=f"{out_path}/class_ens"
    if(model_exist(model_path,start)):
        return "Model exist"
    utils.make_dir(model_path)
    for j in tqdm(range(step)):
        index=start+j
        raw_split=np.load(f"{split_path}/{index}.npz")
        split_j=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        clf_j=clf_factory()
        clf_j,history=split_j.fit_clf(data,clf_j)  
        clf_j.save(f"{model_path}/{index}.keras")

def model_exist(model_path,start):
    if(not os.path.isdir(model_path)):
         return False
    first_path=f"{model_path}/{start}.keras"
    if(not os.path.isfile(first_path)):
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci/vehicle")
    parser.add_argument("--out_path", type=str, default="exp_deep/led7digit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    print(args)
    base_train(data_path=args.data,
               out_path=args.out_path,
               start=args.start,
               step=args.step)