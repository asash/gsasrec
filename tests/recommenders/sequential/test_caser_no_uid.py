import unittest

class TestCaserNoUid(unittest.TestCase):
    def test_caser_model_no_uid(self):
        from aprec.losses.bce import BCELoss
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.models.caser import CaserConfig
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit


        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        caser_config = CaserConfig()
        recommender_config = SequentialRecommenderConfig(caser_config, train_epochs=10,
                                               early_stop_epochs=5, batch_size=5, 
                                               training_time_limit=10, 
                                               loss=BCELoss(), 
                                               sequence_length=5,
                                               use_keras_training=True
                                               )
        recommender = SequentialRecommender(recommender_config)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        USER_ID = '120'
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == '__main__':
    unittest.main()

