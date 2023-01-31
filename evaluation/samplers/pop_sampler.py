from argparse import ArgumentError
import time
import numpy as np
import tqdm
from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.evaluation.evaluation_utils import group_by_user
from aprec.evaluation.metrics.sampled_proxy_metric import SampledProxy
from aprec.evaluation.samplers.sampler import TargetItemSampler

class PooledSamplerWithReplacement(object):
    def __init__(self, items, probs, pool_size=1000000):
        self.items = items
        self.probs = probs
        self.pool_size = pool_size 
        self.reset_pool()

    def reset_pool(self):
        self.pool =  np.random.choice(self.items, self.pool_size, p=self.probs, replace=True)
        self.current_pos = 0

    
    def sample(self, n_items):
        if n_items > self.pool_size:
            raise ArgumentError("can not sample more items than pool size")
        elif self.current_pos + n_items > self.pool_size:
            self.reset_pool()
        old_pos = self.current_pos
        self.current_pos += n_items
        result =  self.pool[old_pos:self.current_pos]
        if len(result) != n_items:
            raise Exception("problem in the code: result length is not equal to requested number")
        return result



class PopTargetItemsSampler(TargetItemSampler):
    def get_sampled_ranking_requests(self):
        items, probs = SampledProxy.all_item_ids_probs(self.actions)
        pooled_sampler = PooledSamplerWithReplacement(items, probs)
        by_user_test = group_by_user(self.test)
        result = []
        for user_id in tqdm.tqdm(by_user_test):
            target_items = set(action.item_id for action in by_user_test[user_id])
            while(len(target_items) < self.target_size):
                item_ids = pooled_sampler.sample(self.target_size - len(target_items))
                target_items = target_items.union(set(item_ids))
            result.append(ItemsRankingRequest(user_id=user_id, item_ids=list(target_items)))
        return result

class PopTargetItemsWithReplacementSampler(TargetItemSampler):
    def get_sampled_ranking_requests(self):
        items, probs = SampledProxy.all_item_ids_probs(self.actions)
        pooled_sampler = PooledSamplerWithReplacement(items, probs)
        by_user_test = group_by_user(self.test)
        result = []
        for user_id in tqdm.tqdm(by_user_test):
            target_items = set(action.item_id for action in by_user_test[user_id])
            sampled_items = []
            while(len(target_items) + len(sampled_items) < self.target_size):
                item_ids = pooled_sampler.sample(self.target_size - len(target_items))
                for item_id in item_ids:
                    if item_id not in target_items:
                        sampled_items.append(item_id) 
            result.append(ItemsRankingRequest(user_id=user_id, item_ids=list(target_items) + sampled_items))
        return result