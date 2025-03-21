import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import os.path
import argparse,json
import base,dataset,ens,utils

def base_train(data_path:str,
               out_path:str,
               ens_type="class_ens",
               start=0,
               step=10):
    path_dict=get_paths(out_path=out_path,
                        ens_type=ens_type,
                        dirs=['models','history','info.js'])
    utils.make_dir(path_dict['ens'])    
    model_paths=get_model_paths(path_dict['models'],start,step)
    print(model_paths)
    if(len(model_paths)==0):
        raise Exception("Models exist")
    utils.make_dir(path_dict['models'])
    utils.make_dir(path_dict['history'])
    data=dataset.read_csv(data_path)
    clf_factory=ens.get_ens(ens_type)
    clf_factory.init(data)
    for index,model_path in tqdm(model_paths):
        raw_split=np.load(f"{path_dict['splits']}/{index}.npz")
        split_j=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
                                       test_index=raw_split["arr_1"])
        clf_j=clf_factory()
        clf_j,history_j=split_j.fit_clf(data,clf_j)  
        hist_dict_j=utils.history_to_dict(history_j)
        clf_j.save(model_path)
        with open(f"{path_dict['history']}/{index}", 'w') as f:
            json.dump(hist_dict_j, f)
    with open(path_dict['info.js'], 'w') as f:
        json.dump(clf_factory.get_info(),f)

def get_paths(out_path,ens_type,dirs):
    ens_path=f"{out_path}/{ens_type}"
    path_dir={dir_i:f"{ens_path}/{dir_i}" 
                    for dir_i in dirs}
    path_dir['ens']=ens_path
    path_dir['splits']=f"{out_path}/splits"
    return path_dir

def get_model_paths(model_path,start,step):
    indexs=[start+j for j in range(step)]
    paths=[ (index,f"{model_path}/{index}.keras") 
               for index in indexs]
    if(not os.path.isdir(model_path)):
        return paths
    paths=[ (i,model_i)
              for i,model_i in paths
                  if(not (os.path.isfile(model_i) or
                         os.path.isdir(model_i)))]
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci/wall-following")
    parser.add_argument("--out_path", type=str, default="new_exp/wine")
    parser.add_argument("--start", type=int, default=28)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--ens_type", type=str, default="separ_class_ens")
    args = parser.parse_args()
    print(args)
    base_train(data_path=args.data,
               out_path=args.out_path,
               start=args.start,
               step=args.step,
               ens_type=args.ens_type)