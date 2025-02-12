import os
import utils

def history_acc(exp_path):
    @utils.DirFun({"in_path":0})
    def helper(in_path):
        for path_i in utils.top_files(in_path):
            history_i=f"{path_i}/history"
            if(os.path.exists(history_i)):
                hist_dict=[utils.read_json(path_j) 
                            for path_j in utils.top_files(history_i)]
                print(hist_dict)
    helper(exp_path)

history_acc("new_exp")     