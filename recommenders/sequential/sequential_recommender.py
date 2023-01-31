import gc
import tempfile
from collections import defaultdict
import tensorflow as tf

from tqdm.auto import tqdm
from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.recommenders.sequential.model_trainier import ModelTrainer
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, SequentialRecsysModel, get_sequential_model
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
import tensorflow as tf
import faiss

class SequentialRecommender(Recommender):
    def __init__(self, config: SequentialRecommenderConfig):
        super().__init__()
        self.config = config
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(list)
        self.metadata = {}
        #we use following two dicts for sampled metrics
        self.item_ranking_requrests = {}
        self.item_ranking_results = {}
        self.model: SequentialRecsysModel = None

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()
            


    def rebuild_model(self):
        self.sort_actions()
        self.pass_parameters()

        model_trainer = ModelTrainer(self)
        train_metadata = model_trainer.train()
        del(model_trainer)
        self.metadata['train_metadata'] = train_metadata

        if self.config.use_ann_for_inference:
                self.build_ann_index()

    def build_ann_index(self):
        embedding_matrix = self.model.get_embedding_matrix().numpy()
        self.index = faiss.IndexFlatIP(embedding_matrix.shape[-1])
        self.index.add(embedding_matrix)
        pass
         
    def pass_parameters(self):
        self.config.loss.set_num_items(self.items.size())
        self.config.train_history_vectorizer.set_sequence_len(self.config.sequence_length)
        self.config.train_history_vectorizer.set_padding_value(self.items.size())
        self.config.pred_history_vectorizer.set_sequence_len(self.config.sequence_length)
        self.config.pred_history_vectorizer.set_padding_value(self.items.size())
        
    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.item_ranking_requrests[request.user_id] = request 
    def get_model(self) -> SequentialRecsysModel:
        data_params = SequentialDataParameters(num_items=self.items.size(),
                                               num_users=self.users.size(),
                                               batch_size=self.config.batch_size, 
                                               sequence_length=self.config.sequence_length)
        return get_sequential_model(self.config.model_config, data_params)

    def recommend(self, user_id, limit, features=None):
        if self.config.use_ann_for_inference:
            model_inputs = self.get_model_inputs(user_id) 
            user_emb = self.model.get_sequence_embeddings([model_inputs]).numpy()
            scores, items = self.index.search(user_emb, limit)
            result = [(self.items.reverse_id(items[0][i]), scores[0][i]) for i in range(len(items[0]))]
        else:    
            scores = self.get_all_item_scores(user_id)
            if user_id in self.item_ranking_requrests:
                self.process_item_ranking_request(user_id, scores)
            best_ids = tf.nn.top_k(scores, limit).indices.numpy()
            result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_item_rankings(self):
        for user_id in self.items_ranking_requests:
            self.process_item_ranking_request(user_id)
        return self.item_ranking_results

    def process_item_ranking_request(self,  user_id, scores=None):
        if (user_id not in self.item_ranking_requrests) or  (user_id in self.item_ranking_results):
            return
        if scores is None:
            scores = self.get_all_item_scores(user_id)
        request = self.item_ranking_requrests[user_id]
        user_result = []
        for item_id in request.item_ids:
            if (self.items.has_item(item_id)) and (self.items.get_id(item_id) < len(scores)):
                user_result.append((item_id, float(scores[self.items.get_id(item_id)])))
            else:
                user_result.append((item_id, float("-inf")))
        user_result.sort(key = lambda x: -x[1])
        self.item_ranking_results[user_id] = user_result
    
    def get_model_inputs(self, user_id, is_val=False):
        if not is_val:
            actions = self.user_actions[self.users.get_id(user_id)]
        else:
            actions = self.user_actions[self.users.get_id(user_id)][:-1]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        session = self.config.pred_history_vectorizer(model_actions)
        session = session.reshape(1, self.config.sequence_length)
        model_inputs = [session]
        return model_inputs
    
    def recommend_multiple(self, recommendation_requets, limit, is_val=False):
        user_ids = [user_id for user_id, features in recommendation_requets]
        model_inputs = list(map(lambda id: self.get_model_inputs(id, is_val)[0], user_ids))
        model_inputs = tf.concat(model_inputs, 0)
        result = []
        if is_val or not(self.config.use_ann_for_inference):
            scoring_func = self.get_scoring_func()
            predictions = scoring_func([model_inputs])
            list(map(self.process_item_ranking_request, user_ids, predictions))
            best_predictions = tf.math.top_k(predictions, k=limit)
            ind = best_predictions.indices.numpy()
            vals = best_predictions.values.numpy()
        else:
            embs =  self.model.get_sequence_embeddings([model_inputs]).numpy()
            vals, ind = self.index.search(embs, limit)
        for i in range(len(user_ids)):
            result.append(list(zip(self.decode_item_ids(ind[i]), vals[i])))
        return result
    
    def get_tensorboard_dir(self):
        if self.tensorboard_dir is not None:
            return self.tensorboard_dir
        else:
            return tempfile.mkdtemp()

    def decode_item_ids(self, ids):
        result = []
        for id in ids:
            result.append(self.items.reverse_id(int(id)))
        return result

    def recommend_batch(self, recommendation_requests, limit, is_val=False, batch_size=None):
        if batch_size is None:
            batch_size = self.config.eval_batch_size
        results = []
        start = 0
        end = min(start + batch_size, len(recommendation_requests))
        print("generating recommendation in batches...")
        pbar = tqdm(total = len(recommendation_requests), ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',  position=0, leave=True, ncols=70)
        while (start < end):
            req = recommendation_requests[start:end]
            results += self.recommend_multiple(req, limit, is_val)
            pbar.update(end - start)
            start = end  
            end = min(start + batch_size, len(recommendation_requests))
            gc.collect()
            tf.keras.backend.clear_session()
        return results

    def get_scoring_func(self):
        if hasattr(self.model, 'score_all_items'):
            return self.model.score_all_items
        else: 
            return self.model

    def get_metadata(self):
        return self.metadata
    
    def get_all_item_scores(self, user_id):
        model_inputs = self.get_model_inputs(user_id) 
        scoring_func = self.get_scoring_func()
        return scoring_func(model_inputs)[0].numpy()
    