import os
import unittest


    

class TestSamplingBert(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def run_bert(self, sampler):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.evaluation.metrics.entropy import Entropy
        from aprec.evaluation.metrics.highest_score import HighestScore
        from aprec.evaluation.metrics.model_confidence import Confidence
        from aprec.recommenders.sequential.models.bert4rec.bert4recft import SampleBERTConfig
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.evaluation.metrics.hit import HIT
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        embedding_size=32
        model_config =  SampleBERTConfig(embedding_size=embedding_size, 
                                         loss='bce',
                                         sampler=sampler)
        recommender_config = SequentialRecommenderConfig(model_config, 
                                                train_epochs=10000, early_stop_epochs=50000,
                                                batch_size=5,
                                                training_time_limit=10, 
                                                sequence_splitter=lambda: ItemsMasking(), 
                                                targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=False),
                                                pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                                eval_batch_size=8,
                                                use_keras_training=True,
                                                extra_val_metrics = [HIT(10), HighestScore(), Confidence('Softmax'), Confidence('Sigmoid'), Entropy('Softmax', 10)])
        
        recommender = SequentialRecommender(recommender_config)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        ranking_request = ItemsRankingRequest('120', ['608', '294', '648'])
        recommender.add_test_items_ranking_request(ranking_request)
        batch1 = [('120', None), ('10', None)]
        recs = recommender.recommender.recommend_multiple(batch1, 10)        
        catalog = get_movies_catalog()
        for rec in recs[0]:
            print(catalog.get_item(rec[0]), "\t", rec[1])

    def test_sampling_bert(self):
       self.run_bert('idf')
   
if __name__ == "__main__":
    unittest.main()
