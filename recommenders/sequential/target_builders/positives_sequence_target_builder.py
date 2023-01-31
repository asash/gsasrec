from random import Random

import numpy as np
from aprec.recommenders.sequential.target_builders.target_builders import TargetBuilder


class PositivesSequenceTargetBuilder(TargetBuilder):
    def __init__(self, sequence_len=64):
        self.random = Random()
        self.sequence_len = sequence_len

    def build(self, user_targets):
        self.targets = []
        for i in range(len(user_targets)):
            targets_for_user = [] 
            seq = user_targets[i]
            if len(seq) < self.sequence_len:
                targets_for_user += [-100.0] * (self.sequence_len - len(seq))
            for target in seq[-self.sequence_len:]:
                targets_for_user.append(target[1])
            self.targets.append(targets_for_user)
        self.targets = np.array(self.targets, 'int64')
    
    def get_targets(self, start, end):
        return [self.targets[start:end]], self.targets[start:end]




