import copy
import gzip
from pathlib import Path
import tempfile
import tensorflow as tf
import ujson
import os
import random
import sys
import time
import traceback
from collections import defaultdict

from tqdm import tqdm

from aprec.evaluation.samplers.sampler import TargetItemSampler
from aprec.utils.os_utils import mkdir_p, shell
from aprec.evaluation.filter_cold_start import filter_cold_start
from aprec.evaluation.evaluation_utils import group_by_user
from multiprocessing_on_dill.context import ForkProcess, ForkContext
from tensorboard import program

def compress(filename):
    print(f"compressing {filename}")
    shell(f"gzip {filename}")
    print(f"done compressing {filename}")

def compress_async(filename):
    eval_process = ForkProcess(target=compress,  args=(filename, ))
    eval_process.start()

    
    

def evaluate_recommender(recommender, test_actions,
                         metrics, out_dir, recommender_name,
                         features_from_test=None,
                         recommendations_limit=900,
                         evaluate_on_samples = False,
                         ):

    print('saving model...')
    try:
        mkdir_p(f"{out_dir}/checkpoints/")
        model_filename = f"{out_dir}/checkpoints/{recommender_name}.dill"
        recommender.save(model_filename)
        compress_async(model_filename)

    except Exception:
        print("Failed saving model...")
        print(traceback.format_exc())
        
    tensorboard_dir = f"{out_dir}/tensorboard/{recommender_name}"
    mkdir_p(tensorboard_dir)
    recommender.set_tensorboard_dir(tensorboard_dir)

    test_actions_by_user = group_by_user(test_actions)
    metric_sum = defaultdict(lambda: 0.0)
    sampled_metric_sum = defaultdict(lambda: 0.0)
    all_user_ids = list(test_actions_by_user.keys())
    requests = []
    for user_id in all_user_ids:
        if features_from_test is not None:
            requests.append((user_id, features_from_test(test_actions)))
        else:
            requests.append((user_id, None))

 
    print("generating predictions...")
    all_predictions = recommender.recommend_batch(requests, recommendations_limit)

    if evaluate_on_samples:
        sampled_rankings = recommender.get_item_rankings()

    print('calculating metrics...')
    user_docs = []
    for i in tqdm(range(len(all_user_ids)), ascii=True):
        user_id = all_user_ids[i]
        predictions = all_predictions[i]
        user_test_actions = test_actions_by_user[user_id]
        user_doc = {"user_id": user_id,
                    "metrics": {},
                    "test_actions": [action.to_json() for action in user_test_actions],
                    "predictions": [(prediction[0], float(prediction[1])) for prediction in predictions],
                    }
        if evaluate_on_samples:
            user_doc["sampled_metrics"] = {}
        for metric in metrics:
            metric_value = metric(predictions, test_actions_by_user[user_id])
            metric_sum[metric.name] += metric_value
            user_doc["metrics"][metric.name] = metric_value
            if evaluate_on_samples:
                sampled_metric_value = metric(sampled_rankings[user_id], test_actions_by_user[user_id])
                sampled_metric_sum[metric.name] += sampled_metric_value
                user_doc["sampled_metrics"][metric.name] = sampled_metric_value

        user_docs.append(user_doc)
    
    mkdir_p(f"{out_dir}/predictions/")
    predictions_filename = f"{out_dir}/predictions/{recommender_name}.json"
    print("saving recommendations...")
    saving_start_time = time.time()
    with open(predictions_filename, "wt") as output:
        ujson.dump(user_docs, output, indent=4)
    saving_time = time.time() - saving_start_time
    print(f"done in {saving_time} seconds ({len(user_docs)/saving_time:0.6} per second)")
    compress_async(predictions_filename)
 
    result = {}
    sampled_result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(test_actions_by_user)
        if evaluate_on_samples:
            sampled_result[metric] = sampled_metric_sum[metric]/len(test_actions_by_user)
    if evaluate_on_samples:
        result["sampled_metrics"] = sampled_result


    return result

class RecommendersEvaluator(object):
    def __init__(self, actions, recommenders, metrics, out_dir, data_splitter,
                 n_val_users, recommendations_limit, callbacks=(),
                 users=None,
                 items=None,
                 experiment_config=None,
                 target_items_sampler: TargetItemSampler = None,
                 remove_cold_start=True, 
                 save_split = False,
                 global_tensorboard_dir = None 
                 ):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders
        self.data_splitter = data_splitter
        self.callbacks = callbacks
        self.out_dir = out_dir
        tensorboard_dir = f"{out_dir}/tensorboard/"
        if global_tensorboard_dir is None:
            global_tensorboard_dir = Path(tempfile.mkdtemp())
        self.global_tensorboard_dir = global_tensorboard_dir
        mkdir_p(tensorboard_dir)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tensorboard_dir, '--host', '0.0.0.0'])
        url = tb.launch()
        print(f"TensorBoard is listening listening on {url}")

        self.features_from_test = None
        self.n_val_users = n_val_users
        print("splitting actions...")
        split_actions_start = time.time()
        self.train, self.test = self.data_splitter(actions)
        split_actions_end = time.time() 
        print(f"actions split in {split_actions_end - split_actions_start} seconds")
        if save_split:
            print(f"saving split for reproducibility purposes...")
            saving_start = time.time()
            self.save_split(self.train, self.test)
            saving_end = time.time()
            print(f"split saved in {saving_end - saving_start} seconds")

        if remove_cold_start:
            self.test = filter_cold_start(self.train, self.test)
        self.users = users
        self.items = items
        all_train_user_ids = list(set([action.user_id for action in self.train]))
        self.recommendations_limit = recommendations_limit
        random.shuffle(all_train_user_ids)
        self.val_user_ids = all_train_user_ids[:self.n_val_users]
        self.sampled_requests = None
        if target_items_sampler is not None:
            print("generating sampled items requests...")
            sampled_requests_generation_start = time.time()
            target_items_sampler.set_actions(self.actions, self.test)
            self.sampled_requests = target_items_sampler.get_sampled_ranking_requests() 
            sampled_requests_generation_end = time.time()
            print(f"sampled requests generated in {sampled_requests_generation_start - sampled_requests_generation_end} seconds")
        self.experiment_config = experiment_config

    def set_features_from_test(self, features_from_test):
        self.features_from_test = features_from_test

    def __call__(self):
        ctx = ForkContext()
        result_queue = ctx.Queue(maxsize=1) 
        result = {}
        result["recommenders"] = {}
        print(f"recommenders to evaluate:")
        for i, recommender_name in enumerate(self.recommenders):
            print(f"{i+1}. {recommender_name}")
        for recommender_name in self.recommenders:
            #using ForkProcess in order to guarantee that every recommender is evaluated in its own process and
            #the recommender releases all resources after evaluating.
            eval_process = ForkProcess(target=self.evaluate_single_recommender,  args=(recommender_name, result_queue))
            eval_process.start()
            eval_process.join()
            if not result_queue.empty(): #successful evaluation:
                evaluation_result = result_queue.get()
                result['recommenders'][recommender_name] = evaluation_result 

                #by some reason logging to tensorboard also wants to use GPU memory.
                # Running this in sandbox in order to release the memomoryt afterwards. 
                log_process = ForkProcess(target=self.log_result_to_tensorboard, args=(evaluation_result,))
                log_process.start()
                log_process.join()
                pass
        return result

    def log_result_to_tensorboard(self, evaluation_result):
        tensorboard_dir = evaluation_result['model_metadata']['tensorboard_dir']
        step = int(evaluation_result['minutes_to_converge'])
        tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        with tf.device("CPU:0"),  tensorboard_writer.as_default(step=step):
            for metric in self.metrics:
                metric_value = float(evaluation_result[metric.name])
                metric_name = f"test/{metric.name}"
                tf.summary.scalar(metric_name, metric_value)
        pass

    def evaluate_single_recommender(self, recommender_name, result_queue):
        try:
            sys.stdout.write("!!!!!!!!!   ")
            print("evaluating {}".format(recommender_name))
            recommender = self.recommenders[recommender_name]()
            tensorboard_dir = f"{self.out_dir}/tensorboard/{recommender_name}"
            mkdir_p(tensorboard_dir)
            tensorboard_run_id = recommender_name + "_" + Path(self.out_dir).name
            os.symlink(os.path.abspath(tensorboard_dir), os.path.abspath((self.global_tensorboard_dir/tensorboard_run_id)))
            recommender.set_out_dir(self.out_dir)
            recommender.set_tensorboard_dir(tensorboard_dir)
            print("adding train actions...")
            for action in tqdm(self.train, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',  position=0, leave=True, ncols=70):
                recommender.add_action(action)
            recommender.set_val_users(self.val_user_ids)
            print("rebuilding model...")
            if self.users is not None:
                print("adding_users")
                for user in self.users:
                    recommender.add_user(user)
            if self.items is not None:
                print("adding items")
                for item in self.items:
                    recommender.add_item(item)
            build_time_start = time.time()
            if self.sampled_requests is not None:
                for request in self.sampled_requests:
                    recommender.add_test_items_ranking_request(request)
            recommender.rebuild_model()
            build_time_end = time.time()
            print("done")
            print("calculating metrics...")
            evaluate_time_start = time.time()
            evaluation_result = evaluate_recommender(recommender, self.test,
                                                     self.metrics, self.out_dir,
                                                     recommender_name, self.features_from_test,
                                                     recommendations_limit=self.recommendations_limit,
                                                     evaluate_on_samples=self.sampled_requests is not None)
            evaluate_time_end = time.time()
            print("calculating metrics...")
            build_time = build_time_end - build_time_start
            evaluation_result['model_build_time'] = build_time 
            evaluation_result['minutes_to_converge'] = self.minutes_to_converge(build_time, recommender.get_metadata()) 
            evaluation_result['model_inference_time'] = evaluate_time_end - evaluate_time_start
            evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
            evaluation_result['model_metadata']['tensorboard_dir'] =  tensorboard_dir
            print("done")

            print(ujson.dumps(evaluation_result))
            result_queue.put(evaluation_result)

            for callback in self.callbacks:
                callback(recommender, recommender_name, evaluation_result, self.experiment_config)
            del (recommender)
        except Exception as ex:
            print(f"ERROR: exception during evaluating {recommender_name}")
            print(ex)
            print(traceback.format_exc())
            try:
                del (recommender)
            except:
                pass

    def minutes_to_converge(self, build_time, model_metadata):
        result_seconds = build_time
        if 'train_metadata' in model_metadata and 'time_to_converge' in model_metadata['train_metadata']:
            result_seconds = model_metadata['train_metadata']['time_to_converge']
        return result_seconds / 60

    def save_split(self, train, test):
        training_actions_saving_start = time.time()
        print("saving train actions...")
        self.save_actions(train, "train.json.gz")
        training_actions_saving_end = time.time()
        print(f"train actions aved in {training_actions_saving_end - training_actions_saving_start} seconds")
        print("saving test actions...")
        test_actions_saving_start = time.time()
        self.save_actions(test, "test.json.gz")
        test_actions_saving_end = time.time()
        print(f"test actions aved in {test_actions_saving_end - test_actions_saving_start} seconds")

    def save_actions(self, actions, filename):
        with open(os.path.join(self.out_dir, filename), 'w') as output:
            for action in tqdm(actions):
                output.write(action.to_json().encode('utf-8') + b"\n")
        compress_async(filename)