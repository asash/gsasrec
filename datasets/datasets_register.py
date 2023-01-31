#This file contains register of all available datasets in our system.
#Unless necessary only use datasets from this file.
import os
import pickle
from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.datasets.booking import get_booking_dataset
from aprec.datasets.movielens100k import get_movielens100k_actions
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.datasets.movielens25m import get_movielens25m_actions
from aprec.datasets.gowalla import get_gowalla_dataset
from aprec.datasets.movies_dataset import get_movies_dataset, get_movies_dataset_with_bands
from aprec.datasets.netflix import get_netflix_dataset
from aprec.datasets.yelp import get_yelp_dataset
from aprec.datasets.amazon import get_amazon_actions
from aprec.datasets.mts_kion import get_mts_kion_dataset
from aprec.datasets.dataset_utils import filter_cold_users, take_user_fraction 
from aprec.utils.os_utils import mkdir_p_local

class DatasetsRegister(object):
    DATA_DIR = "data/cache"

    _all_datasets =  {
        "BERT4rec.ml-1m": lambda: get_bert4rec_dataset("ml-1m"),
        "BERT4rec.steam": lambda: get_bert4rec_dataset("steam"),
        "BERT4rec.beauty": lambda: get_bert4rec_dataset("beauty"),
        "ml-20m": lambda: get_movielens20m_actions(min_rating=0.0),
        "ml-25m": lambda: get_movielens25m_actions(min_rating=0.0),
        "movies_dtaset": lambda: get_movies_dataset(),
        "movies_dtaset_with_budget_band": lambda: get_movies_dataset_with_bands(),
        "ml-100k": lambda: get_movielens100k_actions(min_rating=0.0),
        "booking": lambda: get_booking_dataset(unix_timestamps=True, mark_control=False)[0],
        "gowalla": get_gowalla_dataset,
        "mts_kion": lambda: get_mts_kion_dataset(),
        "yelp": get_yelp_dataset,
        "netflix": get_netflix_dataset,
        "Amazon.Books": lambda: get_amazon_actions("books"),

        "ml-20m_warm5": lambda: filter_cold_users(get_movielens20m_actions(min_rating=0.0), 5), 
        "booking_warm5": lambda: filter_cold_users(get_booking_dataset(unix_timestamps=True, mark_control=False)[0], 5), 
        "gowalla_warm5": lambda: filter_cold_users(get_gowalla_dataset(), 5), 
        "yelp_warm5": lambda: filter_cold_users(get_yelp_dataset(), 5),
        "netflix_warm5": lambda: filter_cold_users(DatasetsRegister.get_from_cache("netflix")(), 5), 
        "mts_kion_warm5": lambda: filter_cold_users(get_mts_kion_dataset(), 5),
        "Amazon.Books_warm5": lambda: filter_cold_users(DatasetsRegister.get_from_cache("Amazon.Books")(), 5),


        "ml-20m_warm10": lambda: filter_cold_users(get_movielens20m_actions(min_rating=0.0), 10), 
        "booking_warm10": lambda: filter_cold_users(get_booking_dataset(unix_timestamps=True, mark_control=False)[0], 10), 
        "gowalla_warm10": lambda: filter_cold_users(get_gowalla_dataset(), 10), 
        "yelp_warm10": lambda: filter_cold_users(get_yelp_dataset(), 10),
        "netflix_warm10": lambda: filter_cold_users(DatasetsRegister.get_from_cache("netflix")(), 10), 


        "ml-20m_warm5_fraction_0.01": lambda: take_user_fraction(DatasetsRegister.get_from_cache("ml-20m_warm5")(), 0.01), 
        "ml-20m_warm5_fraction_0.001": lambda: take_user_fraction(DatasetsRegister.get_from_cache("ml-20m_warm5")(), 0.001), 
        "netflix_fraction_0.001": lambda: take_user_fraction(DatasetsRegister.get_from_cache("netflix")(), 0.001), 
    }
    
    @staticmethod
    def get_dataset_file(dataset_id):
        cache_dir = mkdir_p_local(DatasetsRegister.DATA_DIR)
        dataset_file = os.path.join(cache_dir, dataset_id + ".pickle")
        return dataset_file
 
    @staticmethod
    def get_from_cache(dataset_id):
        dataset_file = DatasetsRegister.get_dataset_file(dataset_id)
        if not os.path.isfile(dataset_file):
            DatasetsRegister.cache_dataset(dataset_file, dataset_id)
        return lambda: DatasetsRegister.read_from_cache(dataset_file)

    @staticmethod 
    def cache_dataset(dataset_file, dataset_id):
        actions =  [action for action in DatasetsRegister._all_datasets[dataset_id]()]
        with open(dataset_file, "wb") as output:
            pickle.dump(actions, output)
    
    @staticmethod
    def read_from_cache(dataset_file):
        with open(dataset_file, "rb") as input:
            actions = pickle.load(input)
        return actions

    def __getitem__(self, item):
        if item not in DatasetsRegister._all_datasets:
            raise KeyError(f"The dataset {item} is not registered")
        return DatasetsRegister.get_from_cache(item)

    def all_datasets(self):
        return list(DatasetsRegister._all_datasets.keys())



