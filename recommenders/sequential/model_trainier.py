from __future__ import annotations
import gc
from typing import TYPE_CHECKING

from collections import defaultdict
import random
import time
from typing import List
import tensorflow as tf
from tqdm import tqdm
from aprec.api.action import Action
from aprec.recommenders.sequential.data_generator.data_generator import DataGenerator, DataGeneratorAsyncFactory, MemmapDataGenerator
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

if TYPE_CHECKING:
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender

class ValidationResult(object):
    def __init__(self, val_loss, val_metric, extra_val_metrics, train_metric, extra_train_metrics, validation_time) -> None:
        self.val_loss = val_loss
        self.val_metric = val_metric
        self.extra_val_metrics = extra_val_metrics
        self.train_metric = train_metric
        self.extra_train_metrics = extra_train_metrics
        self.validation_time = validation_time
       
class TrainingResult(object):
    def __init__(self, training_loss, training_time, trained_batches, trained_samples):
        self.training_loss = training_loss
        self.training_time = training_time
        self.trained_batches = trained_batches
        self.trained_samples = trained_samples

class EpochResult(object):
    def __init__(self, train_result: TrainingResult, val_result: ValidationResult) -> None:
        self.train_result = train_result
        self.val_result = val_result

class ModelTrainer(object):
    def __init__(self, recommender: SequentialRecommender):
        self.recommender = recommender
        tensorboard_dir = self.recommender.get_tensorboard_dir()
        print(f"writing tensorboard logs to {tensorboard_dir}")
        self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.train_users, self.val_users = self.train_val_split()
        
        self.val_recommendation_requets = [(user_id, None) for user_id in self.recommender.val_users]
        self.val_seen, self.val_ground_truth = self.leave_one_out(self.recommender.val_users)

        self.train_users_pool = list(self.recommender.users.straight.keys() - set(self.recommender.val_users))
        self.train_users_sample = random.choices(self.train_users_pool, k=len(self.val_users)) #these users will be used for calculating metric on train
        self.train_sample_recommendation_requests = [(user_id, None) for user_id in self.train_users_sample]
        self.train_sample_seen, self.train_sample_ground_truth = self.leave_one_out(self.train_users_sample)

        self.targets_builder = self.recommender.config.targets_builder()
        self.targets_builder.set_n_items(self.recommender.items.size())
        self.targets_builder.set_train_sequences(self.train_users)
        print("train_users: {}, val_users:{}, items:{}".format(len(self.train_users), len(self.val_users), self.recommender.items.size()))
        self.model_is_compiled = False
        self.recommender.model = self.recommender.get_model()
        self.recommender.model.fit_biases(self.train_users)
        
        if self.recommender.config.val_metric.less_is_better:
            self.best_metric_val = float('inf')
        else:
            self.best_metric_val = float('-inf')

        self.best_val_loss = float('inf')

        self.steps_metric_not_improved = 0
        self.steps_loss_not_improved = 0
        self.best_epoch = -1
        self.best_weights = self.recommender.model.get_weights()
        self.last_no_nan_weights = self.recommender.model.get_weights()
        self.val_generator = self.get_val_generator()
        self.early_stop_flag = False
        self.current_epoch = None 
        self.history: List[EpochResult] = []
        self.trained_samples=0
        self.trained_batches=0
        self.trained_epochs=0
        self.time_to_converge = 0

    def train(self):
        self.start_time = time.time()
        with self.get_train_generator_factory() as train_data_generator_factory:
            for epoch_num in range(self.recommender.config.train_epochs):
                print(f"epoch {epoch_num}")
                self.current_epoch = epoch_num
                train_generator = train_data_generator_factory.next_generator()
                self.epoch(train_generator)
                train_generator.cleanup()
                if self.early_stop_flag:
                    break
            #cleanup loss and optimizer associated with the model
            del(self.recommender.model) 
            self.recommender.model = self.recommender.get_model()
            self.recommender.model.set_weights(self.best_weights)
            
            print(f"taken best model from epoch{self.best_epoch}. best_val_{self.recommender.config.val_metric.name}: {self.best_metric_val}")
            train_metadata = {'time_to_converge': self.time_to_converge}
            return train_metadata
    
    def get_train_generator_factory(self):
        return DataGeneratorAsyncFactory(self.recommender.config,
                                      self.train_users,
                                      self.recommender.items.size(),
                                      targets_builder=self.targets_builder, 
                                      shuffle_data=True)
    def get_val_generator(self):
        return DataGenerator(self.recommender.config, self.val_users, 
                                      self.recommender.items.size(),
                                      targets_builder=self.targets_builder,
                                      shuffle_data=False)


    def epoch(self, generator: MemmapDataGenerator):
        train_result =  self.train_epoch(generator)
        validation_result = self.validate()
        epoch_result = EpochResult(train_result, validation_result)
        self.history.append(epoch_result)
        self.try_update_best_val_metric(epoch_result)
        self.try_update_best_val_loss(epoch_result)
        self.try_early_stop()
        self.log(epoch_result)
        self.epoch_cleanup()

    def epoch_cleanup(self):
        gc.collect()
        tf.keras.backend.clear_session()
        
    def training_time(self):
        return time.time() - self.start_time

    def try_early_stop(self):
        self.steps_to_early_stop = self.recommender.config.early_stop_epochs - self.steps_metric_not_improved
        if self.steps_to_early_stop <= 0:
            print(f"early stopped at epoch {self.current_epoch}")
            self.early_stop_flag = True

        if self.recommender.config.training_time_limit is not None and self.training_time() > self.recommender.config.training_time_limit:
            print(f"time limit stop triggered at epoch {self.current_epoch}")
            self.early_stop_flag = True

    def try_update_best_val_loss(self, epoch_result: EpochResult):
        val_loss = epoch_result.val_result.val_loss
        self.steps_loss_not_improved += 1
        if (val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            self.steps_loss_not_improved = 0

    def try_update_best_val_metric(self, epoch_result: EpochResult):
        val_metric = epoch_result.val_result.val_metric
        self.steps_metric_not_improved += 1
        if (self.recommender.config.val_metric.less_is_better and val_metric < self.best_metric_val) or\
                            (not self.recommender.config.val_metric.less_is_better and val_metric > self.best_metric_val):
            self.steps_metric_not_improved = 0
            self.best_metric_val = val_metric
            self.best_epoch = self.current_epoch
            self.best_weights = self.recommender.model.get_weights()
            self.time_to_converge = self.training_time()

    def log(self, epoch_result: EpochResult):
        self.log_to_console(epoch_result)
        with self.tensorboard_writer.as_default(step=self.trained_samples):
            self.log_to_tensorboard(epoch_result)

    def log_to_tensorboard(self, epoch_result: EpochResult):
        config = self.recommender.config
        validation_result = epoch_result.val_result
        train_result = epoch_result.train_result

        tf.summary.scalar(f"{config.val_metric.name}/val", validation_result.val_metric)
        tf.summary.scalar(f"{config.val_metric.name}/train", validation_result.train_metric)
        tf.summary.scalar(f"{config.val_metric.name}/train_val_diff", validation_result.train_metric - validation_result.val_metric)
        tf.summary.scalar(f"{config.val_metric.name}/best_val", self.best_metric_val)
        tf.summary.scalar(f"{config.val_metric.name}/steps_metric_not_improved", self.steps_metric_not_improved)
        tf.summary.scalar(f"loss/train", train_result.training_loss)
        tf.summary.scalar(f"loss/val", validation_result.val_loss)
        tf.summary.scalar(f"loss/train_val_diff", (train_result.training_loss - validation_result.val_loss))
        tf.summary.scalar(f"loss/evaluations_without_improvement", self.steps_loss_not_improved)
        tf.summary.scalar(f"steps_to_early_stop", self.steps_to_early_stop)
        tf.summary.scalar(f"time/epoch_trainig_time", train_result.training_time)
        tf.summary.scalar(f"time/trainig_time_per_batch", train_result.training_time/train_result.trained_batches)
        tf.summary.scalar(f"time/trainig_time_per_sample", train_result.training_time/train_result.trained_samples)
        tf.summary.scalar(f"time/epoch_validation_time", validation_result.validation_time)
        
        for metric in config.extra_val_metrics:
            tf.summary.scalar(f"{metric.get_name()}/train", validation_result.extra_train_metrics[metric.get_name()])
            tf.summary.scalar(f"{metric.get_name()}/val", validation_result.extra_val_metrics[metric.get_name()])
            tf.summary.scalar(f"{metric.get_name()}/train_val_diff", validation_result.extra_train_metrics[metric.get_name()]
                                                                    - validation_result.extra_val_metrics[metric.get_name()])
        self.recommender.model.log()

    def log_to_console(self, epoch_result: EpochResult):
        config = self.recommender.config
        validation_result = epoch_result.val_result
        train_result = epoch_result.train_result
        print(f"\tval_{config.val_metric.name}: {validation_result.val_metric:.5f}")
        print(f"\tbest_{config.val_metric.name}: {self.best_metric_val:.5f}")
        print(f"\ttrain_{config.val_metric.name}: {validation_result.train_metric:.5f}")
        print(f"\ttrain_loss: {train_result.training_loss}")
        print(f"\tval_loss: {validation_result.val_loss}")
        print(f"\tbest_val_loss: {self.best_val_loss}")
        print(f"\tsteps_metric_not_improved: {self.steps_metric_not_improved}")
        print(f"\tsteps_loss_not_improved: {self.steps_loss_not_improved}")
        print(f"\tsteps_to_stop: {self.steps_to_early_stop}")
        print(f"\ttotal_training_time: {self.training_time()}")
        
    def leave_one_out(self, users):
        seen_result = []
        result = []
        for user_id in users:
            user_actions = self.recommender.user_actions[self.recommender.users.get_id(user_id)]
            seen_items = {self.recommender.items.reverse_id(action[1]) for action in user_actions[:-1]}
            seen_result.append(seen_items)
            last_action = user_actions[-1] 
            user_result = Action(user_id=user_id, item_id=self.recommender.items.reverse_id(last_action[1]), timestamp=last_action[0])
            result.append([user_result])
        return seen_result, result

    def train_val_split(self):
        val_user_ids = [self.recommender.users.get_id(val_user) for val_user in self.recommender.val_users]
        train_user_ids = list(range(self.recommender.users.size()))
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        return train_users, val_users

    # exclude last action for val_users
    def user_actions_by_id_list(self, id_list, val_user_ids=None):
        val_users = set()
        if val_user_ids is not None:
            val_users = set(val_user_ids)
        result = []
        for user_id in id_list:
            if user_id not in val_users:
                result.append(self.recommender.user_actions[user_id])
            else:
                result.append(self.recommender.user_actions[user_id][:-1])
        return result



    def train_epoch(self, generator):
        epoch_training_start = time.time()
        if self.recommender.config.use_keras_training:
            training_loss =  self.train_epoch_keras(generator)
        else:
            training_loss = self.train_epoch_eager(generator)
        epoch_training_time = time.time() - epoch_training_start
        trained_batches = len(generator)
        trained_samples = trained_batches * self.recommender.config.batch_size
        self.trained_batches += trained_batches 
        self.trained_samples += trained_samples 
        self.trained_epochs += 1
        return TrainingResult(training_loss, epoch_training_time, trained_batches, trained_samples)
        

    def train_epoch_keras(self, generator):
        self.ensure_model_is_compiled()
        summary =  self.recommender.model.fit(generator, steps_per_epoch=len(generator))
        return summary.history['loss'][0]

    def ensure_model_is_compiled(self):
        if not self.model_is_compiled:
            self.recommender.model.compile(self.recommender.config.optimizer, self.recommender.config.loss)
            self.model_is_compiled = True
        
    def train_epoch_eager(self, generator):
        pbar = tqdm(generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        variables = self.recommender.model.variables
        loss_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                tape.watch(variables)
                y_pred = self.recommender.model(X, training=True)
                loss_val = tf.reduce_mean(self.recommender.config.loss(y_true, y_pred))
                pass
            grad = tape.gradient(loss_val, variables)
            self.recommender.config.optimizer.apply_gradients(zip(grad, variables))
            loss_sum += loss_val
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}")
        pbar.close()
        train_loss = loss_sum/num_batches
        return train_loss

    def validate(self):
        validation_start = time.time()
        self.val_generator.reset()
        print("validating..")
        pbar = tqdm(self.val_generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        val_loss_sum = 0 
        num_val_batches = 0
        for X, y_true in pbar:
            num_val_batches += 1
            y_pred = self.recommender.model(X, training=True)
            loss_val = tf.reduce_mean(self.recommender.config.loss(y_true, y_pred))
            val_loss_sum += loss_val
        val_loss = val_loss_sum / num_val_batches 

        val_metric, extra_val_metrics = self.get_val_metrics(self.val_recommendation_requets, self.val_seen, self.val_ground_truth, callbacks=True)
        train_metric, extra_train_metrics = self.get_val_metrics(self.train_sample_recommendation_requests, self.train_sample_seen, self.train_sample_ground_truth)
        validation_time = time.time() - validation_start
        return ValidationResult(val_loss, val_metric, extra_val_metrics,
                                train_metric, extra_train_metrics, validation_time=validation_time)

    def get_val_metrics(self, recommendation_requests, seen_items, ground_truth, callbacks=False):
        extra_recs = 0
        if self.recommender.flags.get('filter_seen', False):
            extra_recs += self.recommender.config.sequence_length        
        recs = self.recommender.recommend_batch(recommendation_requests, 
                                                self.recommender.config.val_rec_limit + extra_recs, 
                                                is_val=True, 
                                                batch_size=self.recommender.config.validation_batch_size)
        metric_sum = 0.0
        extra_metric_sums = defaultdict(lambda: 0.0)
        callback_recs, callback_truth = [], [] 
        for rec, seen, truth in zip(recs, seen_items, ground_truth):
            if self.recommender.flags.get('filter_seen', False):
                filtered_rec = [recommended_item for recommended_item in rec if recommended_item[0] not in seen]
                callback_recs.append(filtered_rec)
                callback_truth.append(truth)
                metric_sum += self.recommender.config.val_metric(filtered_rec, truth) 
                for extra_metric in self.recommender.config.extra_val_metrics:
                    extra_metric_sums[extra_metric.get_name()] += extra_metric(filtered_rec, truth)
            else:
                callback_recs.append(rec)
                callback_truth.append(truth)
                metric_sum += self.recommender.config.val_metric(rec, truth) 
                for extra_metric in self.recommender.config.extra_val_metrics:
                    extra_metric_sums[extra_metric.get_name()] += extra_metric(rec, truth)
        if callbacks:
            for callback in self.recommender.config.val_callbacks:
                callback(callback_recs, callback_truth)
                
        val_metric = metric_sum / len(recs)
        extra_metrics = {}
        for extra_metric in self.recommender.config.extra_val_metrics:
            extra_metrics[extra_metric.get_name()] = extra_metric_sums[extra_metric.get_name()] / len(recs)
        return val_metric, extra_metrics



