import os
import random
import tempfile

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
from multiprocessing_on_dill.context import ForkProcess, ForkContext
from aprec.recommenders.sequential.target_builders.target_builders import TargetBuilder
from aprec.utils.os_utils import shell

class DataGenerator(Sequence):
    def __init__(self, config:SequentialRecommenderConfig, user_actions,
                 n_items, targets_builder: TargetBuilder, shuffle_data = True):
        self.config = config
        self.user_actions = user_actions
        self.sequence_lenghth = config.sequence_length
        self.n_items = n_items
        self.sequences_matrix = None
        self.sequence_splitter = config.sequence_splitter()
        self.sequence_splitter.set_num_items(n_items)
        self.sequence_splitter.set_sequence_len(config.sequence_length)
        self.sequence_splitter.set_actions(user_actions)
        self.targets_builder = targets_builder
        self.targets_builder.set_sequence_len(config.sequence_length)
        self.do_shuffle_data = shuffle_data
        self.reset()


    def reset(self):
        if self.do_shuffle_data: 
            self.shuffle_data()
        history, target = self.split_actions(self.user_actions)
        self.sequences_matrix = self.matrix_for_embedding(history)
        self.targets_builder.set_n_items(self.n_items)
        self.targets_builder.build(target)
        self.current_position = 0
        self.max = self.__len__()
    
    def reset_iterator(self):
        self.current_position = 0

    def shuffle_data(self):
        random.shuffle(self.user_actions)

    @staticmethod
    def get_features_matrix(user_features, max_user_features):
        result = []
        for features in user_features:
            result.append([0] * (max_user_features - len(features)) + features)
        return np.array(result)


    def matrix_for_embedding(self, user_actions):
        result = []
        for actions in user_actions:
            result.append(self.config.train_history_vectorizer(actions))
        return np.array(result)

    def build_target_matrix(self, user_targets):
        if self.config.sampled_target is None:
            self.build_full_target_matrix(user_targets)
        else:
            self.build_sampled_targets(user_targets)

    def split_actions(self, user_actions):
        history = []
        target = []
        if self.config.max_batches_per_epoch is not None:
            max_users = self.config.max_batches_per_epoch * self.config.batch_size
        else:
            max_users = len(user_actions)
        for user in user_actions[:max_users]:
            user_history, user_target = self.sequence_splitter.split(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    def __len__(self):
        return self.sequences_matrix.shape[0] // self.config.batch_size

    def __getitem__(self, idx):
        start = idx * self.config.batch_size
        end = (idx + 1) * self.config.batch_size
        history = self.sequences_matrix[start:end]
        model_inputs = [history]
        target_inputs, target = self.targets_builder.get_targets(start, end)
        model_inputs += target_inputs

        return model_inputs, target 

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result


def reverse_positions(session_len, history_size):
    if session_len >= history_size:
        return list(range(history_size, 0, -1))
    else:
        return [0] * (history_size - session_len) + list(range(session_len, 0, -1))

class MemmapDataGenerator(Sequence):
    @staticmethod
    def flush(arr, fname):
        arr = np.array(arr)
        shape = arr.shape
        dtype = arr.dtype
        res = np.memmap(fname, shape=shape, dtype=dtype, mode="write")
        res[:] = arr[:]
        res.flush()
        res._mmap.close()
        del(res)
        return fname, shape, dtype
    
    @staticmethod
    def recover(fname, shape, dtype):
        res = np.memmap(fname, shape=shape, dtype=dtype, mode="readonly")
        return res

    def __init__(self, data_generator, dir):
        self.tempdir = tempfile.mkdtemp(prefix="sequential_train_", dir=dir)
        self.inputs = []
        self.targets = []
        for i in range(len(data_generator)):
            inputs, target = data_generator[i]
            target_name = os.path.join(self.tempdir, f"batch_{i}.target")
            self.targets.append(self.flush(target, target_name))
            mmaped_inputs = []
            for n_input in range(len(inputs)):
                input_name= os.path.join(self.tempdir, f"batch_{i}_input_{n_input}.input")
                mmaped_inputs.append(self.flush(inputs[n_input], input_name))
            self.inputs.append(mmaped_inputs)
        pass
        self.current_position = 0
        self.max = self.__len__()
        self.memmaped_objects = {}

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if idx not in  self.memmaped_objects:
            inputs = []
            for input in self.inputs[idx]:
                inputs.append(self.recover(*input))
            targets = self.recover(*self.targets[idx])
            self.memmaped_objects[idx] = inputs, targets
        return self.memmaped_objects[idx]

    def reset(self):
        self.current_position = 0
        self.max = self.__len__()

    def cleanup(self):
        for idx in list(self.memmaped_objects.keys()):
            inputs, targets = self.memmaped_objects[idx] 
            targets._mmap.close()
            for input in inputs:
                input._mmap.close()
            del(self.memmaped_objects[idx])
        cmd = f"rm -rf {self.tempdir}"
        shell(cmd)

 
class DataGeneratorFactory(object):
    def __init__(self, queue, tempdir, config, *args, **kwargs):
        self.tempdir = tempdir
        self.factory_func = lambda: MemmapDataGenerator(DataGenerator(config, *args, **kwargs), tempdir)
        self.queue = queue
        self.last_generator:MemmapDataGenerator = None

    def __call__(self):
        while True:
            self.last_generator = self.factory_func()
            self.queue.put(self.last_generator)


class DataGeneratorAsyncFactory(object):
    def __init__(self, config: SequentialRecommenderConfig, *args, **kwargs) -> None:
        self.tempdir = tempfile.mkdtemp(prefix = "sequential_recommender_async_factory_")
        self.config = config
        ctx = ForkContext()
        self.result_queue = ctx.Queue(self.config.data_generator_queue_size)
        self.generator_factory = DataGeneratorFactory(self.result_queue, self.tempdir, config, *args, **kwargs)

    def __enter__(self):
        self.processors:List[ForkProcess] = []
        for i in range(self.config.data_generator_processes):
            self.processors.append(ForkProcess(target=self.generator_factory))
            self.processors[-1].daemon = True 
            self.processors[-1].start()
        return self

    def next_generator(self) -> MemmapDataGenerator:
        return self.result_queue.get()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for p in self.processors:
            p.terminate()
            p.join()
        cmd = f"rm -rf {self.tempdir}"
        shell(cmd)

