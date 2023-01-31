from __future__ import annotations
from typing import List, Type

import tensorflow as tf
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel

#https://arxiv.org/abs/1511.06939
class GRU4RecConfig(SequentialModelConfig):
    def __init__(self,
                 output_layer_activation='linear',
                 embedding_size=64,
                 num_gru_layers=3,
                 num_dense_layers=1,
                 activation='relu'):
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.num_gru_layers = num_gru_layers
        self.num_dense_layers = num_dense_layers
        self.activation = activation

    def get_model_architecture(self) -> Type[GRU4Rec]:
        return GRU4Rec 
    
    def as_dict(self) -> dict:
        return self.__dict__


class GRU4Rec(SequentialRecsysModel):
    def __init__(self, model_parameters: GRU4RecConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)    
        self.model_parameters: GRU4RecConfig
        layers = tf.keras.layers
        input = layers.Input(shape=(self.data_parameters.sequence_length))
        x = layers.Embedding(self.data_parameters.num_items + 1, self.model_parameters.embedding_size, dtype='float32')(input)
        for i in range(self.model_parameters.num_gru_layers - 1):
            x = layers.GRU(self.model_parameters.embedding_size, activation=self.model_parameters.activation, return_sequences=True)(x)
        x = layers.GRU(self.model_parameters.embedding_size, activation=self.model_parameters.activation)(x)

        for i in range(self.model_parameters.num_dense_layers):
            x = layers.Dense(self.model_parameters.embedding_size, activation=self.model_parameters.activation)(x)
        output = layers.Dense(self.data_parameters.num_items, name="output", activation=self.model_parameters.output_layer_activation)(x)
        self.model = tf.keras.Model(inputs=[input], outputs=[output], name='GRU')

    @classmethod
    def get_model_config_class(cls) -> Type[GRU4RecConfig]:
        return GRU4RecConfig

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        input = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int32')
        return [input]