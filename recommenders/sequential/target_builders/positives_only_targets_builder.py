import numpy as np
from aprec.recommenders.sequential.target_builders.target_builders import TargetBuilder


class PositvesOnlyTargetBuilder(TargetBuilder):
    def __init__(self, max_targets_per_user = 10):
        self.max_targets_per_user = max_targets_per_user

    def build(self, user_targets):
        result = []
        for i in range(len(user_targets)):
            seq = np.array([item[1] for item in user_targets[i]])
            if len(seq) > self.max_targets_per_user:
                seq = np.random.choice(seq, self.max_targets_per_user)
            if len(seq) < self.max_targets_per_user:
                seq = np.pad(seq, (0,  self.max_targets_per_user - len(seq)), constant_values=self.n_items)
            result.append(seq)
        self.target_matrix = np.array(result)
        pass
        
    def get_targets(self, start, end):
        target_outputs = self.target_matrix[start:end]
        return [target_outputs], target_outputs