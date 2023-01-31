from typing import List
import tensorflow as tf
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.metric import Metric
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.losses.mean_ypred_loss import MeanPredLoss
from aprec.losses.loss import Loss
from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
from aprec.recommenders.sequential.history_vectorizers.history_vectorizer import HistoryVectorizer
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialModelConfig
from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.sequential.targetsplitters.random_fraction_splitter import RandomFractionSplitter


class SequentialRecommenderConfig(object):
    def __init__(self, model_config: SequentialModelConfig,
                 loss: Loss = MeanPredLoss(), #by default we assume that the model is responsible for loss and model.call() returns the loss value which we will optimize. 
                 train_epochs=300, optimizer=tf.keras.optimizers.Adam(),
                 sequence_splitter = RandomFractionSplitter, 
                 targets_builder = FullMatrixTargetsBuilder,
                 batch_size=1000, early_stop_epochs=100, 
                 training_time_limit=None,
                 train_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(), 
                 pred_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(),
                 data_generator_processes = 8, 
                 data_generator_queue_size = 16,
                 max_batches_per_epoch=10,
                 eval_batch_size = 1024, 
                 validation_batch_size=1024,
                 val_rec_limit=40,
                 val_metric = NDCG(10), #Used for early stopping
                 extra_val_metrics: List[Metric] = [], #Used for logging only
                 val_callbacks = [],
                 use_ann_for_inference = False, 
                 sequence_length=200, 
                 use_keras_training=False
                 ):
        self.model_config = model_config
        self.loss = loss
        self.train_epochs = train_epochs
        self.optimizer = optimizer
        self.sequence_splitter = sequence_splitter
        self.targets_builder = targets_builder 
        self.batch_size = batch_size
        self.early_stop_epochs = early_stop_epochs
        self.training_time_limit = training_time_limit
        self.train_history_vectorizer = train_history_vectorizer
        self.pred_history_vectorizer = pred_history_vectorizer
        self.data_generator_processes = data_generator_processes
        self.data_generator_queue_size = data_generator_queue_size
        self.max_batches_per_epoch = max_batches_per_epoch
        self.eval_batch_size = eval_batch_size
        self.val_rec_limit = val_rec_limit
        self.val_metric =  val_metric
        self.extra_val_metrics: List[Metric] = extra_val_metrics
        self.val_callbacks = val_callbacks
        self.use_ann_for_inference = use_ann_for_inference
        self.loss.set_batch_size(self.batch_size)
        self.sequence_length = sequence_length
        self.use_keras_training = use_keras_training 
        self.validation_batch_size = validation_batch_size