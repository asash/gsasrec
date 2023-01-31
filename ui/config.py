from tqdm import tqdm
from aprec.datasets.movielens20m import get_movies_catalog, get_movielens20m_actions
from aprec.recommenders.mlp_historical import GreedyMLPHistorical
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
import sys

CATALOG = get_movies_catalog()

actions = get_movielens20m_actions(1.0)
recommender = FilterSeenRecommender(GreedyMLPHistorical(train_epochs=300))

cnt = 0
for action in tqdm(actions, ascii=True):
    recommender.add_action(action)
    cnt += 1

sys.stderr.write("building model...")

recommender.rebuild_model()

sys.stderr.write("ready.")

RECOMMENDER = recommender
