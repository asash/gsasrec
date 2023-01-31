from __future__ import annotations
from typing import List, Type
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
import tensorflow as tf

#https://dl.acm.org/doi/abs/10.1145/3159652.3159656
#This is a simplified version of Caser model, which doesn't use user embeddings
#We assume that user embedding is not available for in the sequential recommendation case

class CaserConfig(SequentialModelConfig):
    def __init__(self,
                 output_layer_activation='linear', embedding_size=64,
                 n_vertical_filters=4, n_horizontal_filters=16,
                 dropout_ratio=0.5, activation='relu'):

        self.output_layer_activation = output_layer_activation
        self.embedding_size=embedding_size
        self.n_vertical_filters = n_vertical_filters
        self.n_horizontal_filters = n_horizontal_filters
        self.dropout_ratio = dropout_ratio
        self.activation = activation
    
    def get_model_architecture(self) -> Type[Caser]:
        return Caser 
    
    def as_dict(self) -> dict:
        return self.__dict__


class Caser(SequentialRecsysModel):
    @classmethod
    def get_model_config_class(cls) -> Type[CaserConfig]:
        return CaserConfig

    def __init__(self, model_parameters: CaserConfig,
                 data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)    
        self.model_parameters: CaserConfig 
        layers = tf.keras.layers

        input = layers.Input(shape=(self.data_parameters.sequence_length))
        model_inputs = [input]
        x = layers.Embedding(self.data_parameters.num_items + 1, self.model_parameters.embedding_size, dtype='float32')(input)
        x = layers.Reshape(target_shape=(self.data_parameters.sequence_length, self.model_parameters.embedding_size, 1))(x)
        vertical = layers.Convolution2D(self.model_parameters.n_vertical_filters, kernel_size=(self.data_parameters.sequence_length, 1),
                                        activation=self.model_parameters.activation)(x)
        vertical = layers.Flatten()(vertical)
        horizontals = []
        for i in range(self.data_parameters.sequence_length):
            horizontal_conv_size = i + 1
            horizontal_convolution = layers.Convolution2D(self.model_parameters.n_horizontal_filters,
                                                          kernel_size=(horizontal_conv_size,
                                                                       self.model_parameters.embedding_size), strides=(1, 1),
                                                          activation=self.model_parameters.activation)(x)
            pooled_convolution = layers.MaxPool2D(pool_size=(self.data_parameters.sequence_length - horizontal_conv_size + 1, 1)) \
                (horizontal_convolution)
            pooled_convolution = layers.Flatten()(pooled_convolution)
            horizontals.append(pooled_convolution)
        x = layers.Concatenate()([vertical] + horizontals)
        x = layers.Dropout(self.model_parameters.dropout_ratio)(x)
        x = layers.Dense(self.model_parameters.embedding_size, activation=self.model_parameters.activation)(x)

        output = layers.Dense(self.data_parameters.num_items, activation=self.model_parameters.output_layer_activation)(x)
        self.model = tf.keras.Model(model_inputs, outputs=output)

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        input = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int32')
        return [input]
