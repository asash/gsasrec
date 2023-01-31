from ast import List
from collections import Counter
import copy
from typing import Dict

import numpy as np

from aprec.recommenders.sequential.targetsplitters.targetsplitter import TargetSplitter

class FairItemMasking(TargetSplitter):
    def __init__(self, masking_prob = 0.2,
                 max_predictions_per_seq = 20
                 
                 ) -> None:
        super().__init__()
        self.masking_prob = masking_prob
        self.max_predictions_per_seq = max_predictions_per_seq

    
    def set_item_attributes(self, item_attributes: List[int]):
        self.item_attributes = item_attributes
        self.temperature = np.ones(len(self.item_attributes))
        
        
    def set_actions(self, actions):
        attribute_counts = Counter()
        for action in actions:
            attribute_counts[action.item_id]
            
        
        

    def split(self, sequence):
        seq = sequence[-self.seqence_len: ]
        seq_len = len(seq)

        if len(seq) < self.seqence_len:
            seq = [(-1, self.num_items)] * (self.seqence_len - len(seq)) + seq

        if not self.force_last:
            n_masks = min(self.max_predictions_per_seq,
                            max(1, int(round(len(sequence) * self.masking_prob))))
            sample_range = list(range(len(seq) - seq_len, len(seq)))
            rss_vals = np.array([self.recency_importance(self.seqence_len, pos) for pos in sample_range])
            rss_vals_sum = np.sum(rss_vals)
            probs = rss_vals / rss_vals_sum
            mask_positions = self.random.choice(sample_range, n_masks, p=probs, replace=False)
        else:
            n_masks = 1
            mask_positions = [len(seq) - 1]
        train = copy.deepcopy(seq)
        labels = []
        mask_token = self.num_items + 1 #self.num_items is used for padding
        for position in mask_positions:
            labels.append((position, seq[position]))
            train[position] = (train[position][0], mask_token)
        return train, (seq_len, labels)
    