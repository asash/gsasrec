import math
import os
import numpy as np
import pandas as pd
from aprec.api.action import Action
from tqdm import tqdm

from aprec.utils.os_utils import get_dir, shell


MOVIES_DATASET_RAW_FILE = "movies_dataset.zip"
DATA_DIR = "data/movies_dataset"
MOVIES_URL = "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download"
MOVIES_DATA_DIR = os.path.join(get_dir(), DATA_DIR)
RATINGS_FILE = "ratings.csv"

def get_movies_dataset_file():
    full_filename = os.path.abspath(os.path.join(MOVIES_DATA_DIR, MOVIES_DATASET_RAW_FILE))
    if not (os.path.isfile(full_filename)):
        raise Exception(f"We do not support automatic download for Movies Dataset\n" +
                            f"Please download it manually from {MOVIES_URL} and put it into {MOVIES_DATA_DIR}")
    return full_filename


def preprocess():
    dataset_file = get_movies_dataset_file()
    ratings_file = os.path.join(os.path.abspath(MOVIES_DATA_DIR), RATINGS_FILE)
    if not (os.path.isfile(ratings_file)):
        shell("unzip -o {} -d {}".format(dataset_file, os.path.abspath(MOVIES_DATA_DIR)))
    return ratings_file

def get_movies_dataset():
    ratings_file = preprocess()
    data = pd.read_csv(ratings_file).to_numpy()
    actions = []
    for i in tqdm(range(len(data))):
       action = Action(user_id = str(int(data[i][0])), item_id = str(int(data[i][1])), timestamp=data[i][3], data={'rating': data[i][2]}) 
       actions.append(action)
    return actions

def get_mapping_tmdb():
    preprocess()
    tmdb_to_movie = {}
    links = pd.read_csv(os.path.join(MOVIES_DATA_DIR, 'links.csv'))
    for id, row in links.iterrows():
        if not(math.isnan(row['tmdbId']) or math.isnan(row['movieId'])):
            tmdb_to_movie[str(int(row['tmdbId']))] = str(int(row['movieId']))
    return tmdb_to_movie

def convert_to_int(s):
    try:
        return int(s)
    except:
        return 0

def get_movies_budget_bands():
    preprocess()
    mapping = get_mapping_tmdb()
    df = pd.read_csv(os.path.join(MOVIES_DATA_DIR, 'movies_metadata.csv'))
    df.budget = df.budget.apply(convert_to_int)
    non_zero_movies = df[df.budget != 0]
    non_zero_movies_budgets = list(non_zero_movies['budget'])
    percentiles = []
    for i in range(10, 101, 10):
        percentiles.append(np.percentile(non_zero_movies_budgets, i))
    def get_band(budget):
        for i in range(len(percentiles)):
            if budget < percentiles[i]:
                return i
        return len(percentiles) -1
    non_zero_movies['band'] = non_zero_movies['budget'].apply(get_band)
    result = {}
    for id, row in non_zero_movies.iterrows():
        if str(row['id']) in mapping:
            result[mapping[str(row['id'])]] = row['band']
    return result

def get_movies_dataset_with_bands():
    movies_with_bands = get_movies_budget_bands()
    actions = get_movies_dataset()
    result = [] 
    for action in actions:
        if action.item_id not in movies_with_bands:
            continue
        action.data['budget_band'] = movies_with_bands[action.item_id]
        result.append(action)
    return result
