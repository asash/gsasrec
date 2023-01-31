from __future__ import annotations
from typing import List, Type
import numpy as np
import tensorflow as tf
from aprec.losses.get_loss import listwise_loss_from_config
from aprec.recommenders.sequential.samplers.sampler import get_negatives_sampler

from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from transformers import TFBertMainLayer, BertConfig

class SampleBERTConfig(SequentialModelConfig):
    def __init__(self, 
                 embedding_size = 64, 
                 attention_probs_dropout_prob = 0.2,
                 hidden_act = "gelu",
                 hidden_dropout_prob = 0.2,
                 initializer_range = 0.02,
                 intermediate_size = 128,
                 num_attention_heads = 2,
                 num_hidden_layers = 3,
                 type_vocab_size = 2, 
                 output_layer_activation = 'linear',
                 num_negative_samples = 200,
                 loss = 'bce',
                 loss_parameters = {},
                 sampler='random'
                ):
        self.embedding_size = embedding_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads 
        self.num_hidden_layers = num_hidden_layers 
        self.type_vocab_size = type_vocab_size      
        self.output_layer_activation = output_layer_activation
        self.loss = loss
        self.loss_parameters = loss_parameters 
        self.num_negative_samples = num_negative_samples
        self.sampler = sampler

    def as_dict(self) -> dict:
        return self.__dict__

    def get_model_architecture(self) -> Type[BERT4RecFTModel]:
        return BERT4RecFTModel

class BERT4RecFTModel(SequentialRecsysModel):
    def __init__(self, model_parameters: SampleBERTConfig, data_parameters: SequentialDataParameters,
                        *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: SampleBERTConfig
        self.bert_config = BertConfig(
            vocab_size = self.data_parameters.num_items + 3, # +1 for mask item, +1 for padding, +1 for ignore_item
            hidden_size = self.model_parameters.embedding_size,
            max_position_embeddings=2*self.data_parameters.sequence_length, 
            attention_probs_dropout_prob=self.model_parameters.attention_probs_dropout_prob, 
            hidden_act=self.model_parameters.hidden_act, 
            hidden_dropout_prob=self.model_parameters.hidden_dropout_prob, 
            initializer_range=self.model_parameters.initializer_range, 
            num_attention_heads=self.model_parameters.num_attention_heads, 
            num_hidden_layers=self.model_parameters.num_hidden_layers, 
            type_vocab_size=self.model_parameters.type_vocab_size, 
        )
        self.bert = TFBertMainLayer(self.bert_config, False)
        self.loss_ = listwise_loss_from_config(self.model_parameters.loss, self.model_parameters.loss_parameters)
        self.loss_.set_num_items(self.model_parameters.num_negative_samples + 1)
        self.loss_.set_batch_size(self.data_parameters.batch_size*self.data_parameters.sequence_length)
        self.output_layer_activation = tf.keras.activations.get(self.model_parameters.output_layer_activation)
        self.position_ids_for_pred = tf.expand_dims(tf.range(1, self.data_parameters.sequence_length+1), 0) 
        self.sampler = get_negatives_sampler(self.model_parameters.sampler, self.data_parameters, self.model_parameters.num_negative_samples)

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        masked_sequences = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
        labels_masked = tf.cast(tf.fill((self.data_parameters.batch_size, self.data_parameters.sequence_length-1), -100), 'int64')
        labels_non_masked = tf.zeros((self.data_parameters.batch_size, 1), 'int64')
        labels = tf.concat([labels_masked, labels_non_masked], -1)
        positions = tf.zeros_like(masked_sequences)
        return [masked_sequences, labels, positions]
    
    @classmethod
    def get_model_config_class(cls) -> Type[SampleBERTConfig]:
        return SampleBERTConfig

    def fit_biases(self, train_users):
        self.sampler.fit(train_users)

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        labels = inputs[1]
        negatives = self.sampler(sequences, labels)
        candidates = tf.concat([tf.expand_dims(tf.nn.relu(labels), -1), negatives], -1)
        positions = inputs[2]
        bert_output = self.bert(sequences, position_ids=positions)              
        emb_matrix = tf.gather(self.bert.embeddings.weight, candidates)
        result = tf.einsum("ijk,ijmk->ijm", bert_output[0], emb_matrix)
        logits = self.output_layer_activation(result)

        ground_truth_positives = tf.ones((self.data_parameters.batch_size, self.data_parameters.sequence_length, 1))
        ground_truth_negatives = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length,
                                          self.model_parameters.num_negative_samples))
        ground_truth = tf.concat([ground_truth_positives, ground_truth_negatives], -1) 
        use_mask = tf.expand_dims(tf.cast(labels != -100, 'float32'), -1)
        ground_truth_masked = use_mask * ground_truth +  -100 * (1-use_mask) 
        return self.get_loss(ground_truth_masked, logits)

    def get_loss(self, ground_truth, logits):
        num_lists = self.data_parameters.batch_size * self.data_parameters.sequence_length
        items_per_list = self.model_parameters.num_negative_samples + 1
        ground_truth = tf.reshape(ground_truth, (num_lists, items_per_list))
        logits = tf.reshape(logits, (num_lists, items_per_list))
        return self.loss_.loss_per_list(ground_truth, logits)
    
    def get_embedding_matrix(self):
        return self.bert.embeddings.weight[:-3,:]
    
    def get_sequence_embeddings(self, inputs):
        sequence = inputs[0] 
        bert_output = self.bert(sequence, position_ids=self.position_ids_for_pred)[0][:,-1,:]             
        return bert_output
 

    def score_all_items(self, inputs): 
        embedding_matrix = self.get_embedding_matrix()
        sequence_embeddings = self.get_sequence_embeddings(inputs)
        result = tf.einsum("ij,kj->ki", embedding_matrix, sequence_embeddings)
        return result
    