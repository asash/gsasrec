from collections import defaultdict
from pathlib import PosixPath
from typing import List

import numpy as np
from aprec.api.action import Action
from aprec.utils.item_id import ItemId
from aprec.utils.os_utils import mkdir_p


class MapedSequences(object):
    ALL_SEQUENCES = 'all_sequences.mmap'
    BORDERS = 'borders.mmap'

    def __init__(self, directory, n_users, n_items):
        self.directory = directory
        self.is_maped = False
        
        self.sequences = None
        self.borders = None
        
    @staticmethod
    def build(user_actions, n_users, n_items, directory:PosixPath):
        all_sequences = []
        borders = []
        for i in range(n_users):
            user_sequence = []
            for action in user_actions[i]:
                item = action[0]
                user_sequence.append(item)
            all_sequences += user_sequence
            borders.append(len(all_sequences))
        all_sequences = np.array(all_sequences, dtype='int32')
        sequences_map = np.memmap(directory/MapedSequences.ALL_SEQUENCES, shape=all_sequences.shape, dtype='int32', mode="write")
        sequences_map[:] = all_sequences[:]
        sequences_map.flush()
        borders = np.array(borders, dtype='int32')
        borders_map = np.memmap(directory/MapedSequences.ALL_SEQUENCES, shape=all_sequences.shape, dtype='int32', mode="write")
        borders_map.flush()

class SequentialDataset(object):
    def __init__(self):
        self.user_mapping = ItemId()
        self.item_mapping = ItemId()
        self.user_actions = defaultdict(list)
        self.is_sorted = True
        
    def add_action(self, action):
        user_id = self.user_mapping.get_id(action.user_id)    
        item_id = self.item_mapping.get_id(action.item_id)

        if self.user_actions.has(user_id) and self.user_actions[user_id][-1].timestamp > action.timestamp:
            self.is_sorted = False

        self.user_actions[user_id].append((item_id, action.timestamp))
        
    def sort(self):
        if not self.is_sorted:
            for user in self.user_actions:
                self.user_actions[user].sort(lambda a: a.timestamp)
            self.is_sorted = True

    
    