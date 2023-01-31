import unittest
from multiprocessing import Process, Pipe
from aprec.recommenders.recommender import Recommender

def train_model(conn):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
    from aprec.recommenders.sequential.targetsplitters.last_item_splitter import SequenceContinuation
    from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder


    import tempfile
    from aprec.datasets.movielens20m import get_movielens20m_actions
    from aprec.utils.generator_limit import generator_limit
    USER_ID = '120'

    val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    model_config = SASRecConfig(loss='softmax_ce')
    recommender_config = SequentialRecommenderConfig(model_config, 
                                              train_epochs=10000, early_stop_epochs=50000,
                                              batch_size=5,
                                              targets_builder=PositvesOnlyTargetBuilder,
                                              training_time_limit=3, sequence_splitter=SequenceContinuation, 
                                              use_keras_training=True)
    recommender = SequentialRecommender(recommender_config)
    recommender.set_val_users(val_users)
    for action in generator_limit(get_movielens20m_actions(), 100000):
         recommender.add_action(action)
    recommender.rebuild_model()
    checkpoint_file = tempfile.NamedTemporaryFile(suffix='.dill', delete=False).name 
    recommender.save(checkpoint_file) 
    recs = recommender.recommend(USER_ID, 10)
    conn.send((checkpoint_file, recs, ))



class TestSasrecModel(unittest.TestCase):
    def test_sasrec_model_saving(self):
        #train model in a separate process, avoid data sharing and garbage in tf session 
        parent_conn, child_conn = Pipe()
        p = Process(target=train_model, args=(child_conn, ))
        p.start()
        checkpoint, recs = parent_conn.recv()
        p.join()
        print(checkpoint)
        print(recs)

        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        recovered_recommender = SequentialRecommender.load(checkpoint)
        recommendations_from_recovered = recovered_recommender.recommend('120', 10)
        self.assertEqual(recs, recommendations_from_recovered)



if __name__ == "__main__":
    unittest.main()
    
