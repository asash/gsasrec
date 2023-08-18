from __future__ import annotations

from typing import List, Type
import numpy as np
import tensorflow as tf
from aprec.losses.get_loss import listwise_loss_from_config
from aprec.losses.loss import ListWiseLoss

from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialModelConfig, SequentialRecsysModel
from transformers import BertConfig, TFBertMainLayer, TFBertForMaskedLM

NUM_SPECIAL_ITEMS = 3 # +1 for mask item, +1 for padding, +1 for ignore_item

class FullBERTConfig(SequentialModelConfig):
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
                 loss = 'softmax_ce',
                 loss_parameters = {},
                 num_samples_normalization=False,
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
        self.loss = loss
        self.num_samples_normalization = num_samples_normalization
        self.loss_parameters = loss_parameters
        
    def as_dict(self):
        return self.__dict__
    
    def get_model_architecture(self) -> Type[FullBertModel]:
        return FullBertModel
        


class FullBertModel(SequentialRecsysModel):
    def __init__(self, model_parameters: FullBERTConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(model_parameters, data_parameters, *args, **kwargs)
        self.model_parameters: FullBERTConfig
        bert_config = BertConfig(
            vocab_size = self.data_parameters.num_items + NUM_SPECIAL_ITEMS, 
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
        self.num_items = bert_config.vocab_size - NUM_SPECIAL_ITEMS 
        self.token_type_ids = tf.constant(tf.zeros(shape=(self.data_parameters.batch_size, bert_config.max_position_embeddings)))
        self.bert = TFBertForMaskedLM(bert_config)
        self.loss_ = listwise_loss_from_config(self.model_parameters.loss, self.model_parameters.loss_parameters)
        self.loss_.set_num_items(self.num_items)
        self.loss_.set_batch_size(self.data_parameters.batch_size*self.data_parameters.sequence_length)
        self.position_ids_for_pred = tf.constant(np.array(list(range(1, self.data_parameters.sequence_length +1))).reshape(1, self.data_parameters.sequence_length))
        self.num_samples_normalization = self.model_parameters.num_samples_normalization
        self.sequence_length = self.data_parameters.sequence_length

    @classmethod
    def get_model_config_class(cls) -> Type[FullBERTConfig]:
        return FullBERTConfig

    def get_dummy_inputs(self) -> List[tf.Tensor]:
        masked_sequences = tf.zeros((self.data_parameters.batch_size, self.data_parameters.sequence_length), 'int64')
        labels = tf.zeros_like(masked_sequences)
        positions = tf.zeros_like(masked_sequences)
        return [masked_sequences, labels, positions]

    def call(self, inputs, **kwargs):
        masked_sequences = inputs[0]
        labels = inputs[1]
        positions = inputs[2]
        train = kwargs.get('training', False)
        batch_size = self.data_parameters.batch_size
        positive_idx = tf.expand_dims(tf.nn.relu(labels), -1) #avoid boundary problems, negative values will be filteret later anyway
        sample_num = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, batch_size, dtype='int64'), -1), [1, self.sequence_length]), -1)
        sequence_pos = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, self.sequence_length, dtype='int64'), 0), [batch_size, 1]), -1)
        indices = tf.concat([sample_num, sequence_pos, positive_idx], -1)
        values = tf.ones([batch_size, self.sequence_length])
        use_mask = tf.tile(tf.expand_dims(tf.cast(labels!=-100,'float32'), -1),[1, 1, self.num_items + NUM_SPECIAL_ITEMS])
        ground_truth = tf.scatter_nd(indices, values, [batch_size, self.sequence_length, self.num_items + NUM_SPECIAL_ITEMS])
        ground_truth = use_mask*ground_truth + -100 * (1-use_mask)
        bert_output = self.bert(masked_sequences, position_ids = positions, return_dict=True, output_hidden_states=True, labels=labels)
        hidden_states = self.bert.mlm.predictions.transform(bert_output.hidden_states[-1])
        embeddings = self.bert.bert.embeddings.weight
        logits = tf.einsum("bse, ne -> bsn", hidden_states, embeddings)
        logits = tf.nn.bias_add(logits, self.bert.mlm.predictions.bias)
        loss=self.get_loss(ground_truth,logits)
        return loss

    def get_loss(self, ground_truth, logits):
        num_masked_samples = tf.reduce_sum(tf.cast(ground_truth[:,:,0] != -100, 'float32'), -1)
        num_lists = self.data_parameters.batch_size * self.data_parameters.sequence_length
        items_per_list = self.data_parameters.num_items + NUM_SPECIAL_ITEMS
        ground_truth = tf.reshape(ground_truth, (num_lists, items_per_list))
        logits = tf.reshape(logits, (num_lists, items_per_list))
        if self.num_samples_normalization:
            sample_weights = tf.expand_dims(tf.expand_dims(tf.math.divide_no_nan(1.0, num_masked_samples), -1), -1)
            sample_weights = tf.tile(sample_weights, [1, self.data_parameters.sequence_length, items_per_list])
            sample_weights = tf.reshape(sample_weights, (num_lists, items_per_list))
            return self.loss_.loss_per_list(ground_truth, logits, sample_weights)
        else:
            return self.loss_.loss_per_list(ground_truth, logits)
   
    def score_all_items(self, inputs): 
        sequence = inputs[0] 
        bert_output  = self.bert(sequence, position_ids=self.position_ids_for_pred, return_dict=True, output_hidden_states=True)
        hidden_states = self.bert.mlm.predictions.transform(bert_output.hidden_states[-1])
        embeddings = self.bert.bert.embeddings.weight
        logits = tf.einsum("bse, ne -> bsn", hidden_states, embeddings)
        logits = tf.nn.bias_add(logits, self.bert.mlm.predictions.bias)
        return logits[:, -1,:-NUM_SPECIAL_ITEMS]
