from __future__ import annotations 
from typing import List, Type
import tensorflow as tf

class SequentialModelConfig(object):
    def __init__(self):
        self.config = {}
    
    def as_dict(self) -> dict:
        return self.config 
    
    def get_model_architecture(self) -> Type[SequentialRecsysModel]:
        raise NotImplementedError()
    

class SequentialDataParameters(object):
    def __init__(self, num_users, num_items, sequence_length, batch_size) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.batch_size = batch_size
    
    def as_dict(self):
        return self.__dict__

class SequentialRecsysModel(tf.keras.Model):
    @classmethod
    def get_model_config_class() -> Type[SequentialModelConfig]:
        raise NotImplementedError()

    def __init__(self, model_parameters: SequentialModelConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_parameters = model_parameters
        self.data_parameters = data_parameters

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        raise NotImplementedError()

    def fit_biases(self, train_users):
        pass        

    #write tensorboard staff metrics here
    def log(self):
        pass

    @classmethod
    def from_config(cls, config: dict):
        data_parameters = SequentialDataParameters(**config['data_parameters'])
        model_parameters = cls.get_model_config_class()(**config['model_parameters'])
        model = cls(model_parameters, data_parameters)
        dummy_data = model.get_dummy_inputs() 
        model(dummy_data, training=False) #dummy call to build the model 
        return model
    
    def get_config(self):
        return get_config_dict(self.model_parameters, self.data_parameters)

    

def get_sequential_model(model_config: SequentialModelConfig, data_parameters: SequentialDataParameters):
    config = get_config_dict(model_config, data_parameters)
    model_arch = model_config.get_model_architecture()
    return model_arch.from_config(config)

def get_config_dict(model_config, data_parameters):
    model_config_dict = model_config.as_dict()
    data_parameters_dict = data_parameters.as_dict()
    config = {'model_parameters': model_config_dict, 'data_parameters': data_parameters_dict}
    return config


