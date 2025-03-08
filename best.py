import numpy as np
import pandas as pd 
from collections import namedtuple
from collections import defaultdict
import base

def exp(in_path):
    data_split=base.get_splits(data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1,
                                    split_type="unaggr")
#    utils.make_dir(split_path)
    for i,split_i in enumerate(data_split.splits):
        print(split_i)

in_path="../uci/cleveland"
exp(in_path)