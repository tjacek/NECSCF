import utils
utils.silence_warnings()
import numpy as np
import argparse
import base

def gen_splits(data_path,
               out_path,
               n_splits=10,
               n_repeats=10):
    utils.make_dir(out_path)
    @utils.DirFun({"in_path":0,"out_path":1})
    def helper(in_path,out_path):
        utils.make_dir(out_path)
        data_split=base.get_splits(data_path=in_path,
                                   n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   split_type="unaggr")
        split_path=f"{out_path}/splits"
        utils.make_dir(split_path)
        for i,split_i in enumerate(data_split.splits):
            split_i.save(f"{split_path}/{i}")
    helper(data_path,out_path)

if __name__ == '__main__':
    gen_splits(data_path="../uci",
               out_path="exp_deep",
               n_splits=10,
               n_repeats=10)