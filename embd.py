import tensorflow as tf
import deep#ens

class NECSCF(object):
    def __init__(self,model):
        self.model=model	

def read_models(in_path,
                ens_type="class_ens",
                start=0,
                step=10):
    split_path=f"{in_path}/splits"
    model_path=f"{in_path}/{ens_type}/models"
#    ens_factory
    for index in range(step):
        i=start+index
        model_path_i=f"{model_path}/{i}.keras"
        print(model_path_i)
        model_i=tf.keras.models.load_model(model_path_i,
                                           custom_objects={"loss":deep.WeightedLoss()})
        yield NECSCF(model)

read_models("new_exp/cmc")