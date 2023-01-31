import unittest
from aprec.datasets.movielens20m import get_movies_catalog
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder

from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling, exponential_importance

def dnn(model_config, sequence_splitter, 
                target_builder,
                training_time_limit=3,  
                max_epochs=10000, 
                pred_history_vectorizer = DefaultHistoryVectrizer()
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender, SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,
                                        train_epochs=max_epochs,
                                        early_stop_epochs=100,
                                        batch_size=5,
                                        training_time_limit=training_time_limit,
                                        sequence_splitter=sequence_splitter, 
                                        targets_builder=target_builder, 
                                        use_keras_training=True,
                                        extra_val_metrics=[HighestScore()],
                                        pred_history_vectorizer=pred_history_vectorizer) 
    return SequentialRecommender(config)

def sasrec_rss(recency_importance, add_cls=False):
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance), add_cls=add_cls)
        pred_history_vectorizer = AddMaskHistoryVectorizer() if add_cls else DefaultHistoryVectrizer()
        sasrec_config = SASRecConfig()
        return dnn(
                sasrec_config, 
                sequence_splitter=target_splitter,
                target_builder=PositvesOnlyTargetBuilder, 
            pred_history_vectorizer=pred_history_vectorizer
            )

class TestSasrecRss(unittest.TestCase):
    def test_sasrec_model_sampled_target_cls(self):
        self.run_model(True)

    def run_model(self, add_cls):
        USER_ID='10'
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = sasrec_rss(0.8, add_cls) 
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

        

if __name__ == "__main__":
    unittest.main()
