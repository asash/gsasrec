import unittest


class TestSasrecNoEmbeddingReuse(unittest.TestCase):
    def test_sasrec_model_no_reuse(self):
        from aprec.recommenders.sequential.targetsplitters.last_item_splitter import SequenceContinuation
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        sasrec_config = SASRecConfig(embedding_size=32, reuse_item_embeddings=False)
        recommender_config = SequentialRecommenderConfig(sasrec_config, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=5,  
                                               use_keras_training=True,
                                               max_batches_per_epoch=100,
                                               sequence_splitter=SequenceContinuation,
                                               sequence_length=5,
                                               targets_builder=PositvesOnlyTargetBuilder, 
                                               )

   
        recommender = SequentialRecommender(recommender_config)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        USER_ID='120'
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()