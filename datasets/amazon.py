from aprec.api.action import Action
from aprec.datasets.download_file import download_file


URLS = {"books": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv"}
DATA_DIR = "data/amazon"

def download(category):
    filename = download_file(URLS[category], f"{category}.csv", DATA_DIR)
    return filename

def get_amazon_actions(category):
    filename = download(category)
    result = []
    for line in open(filename):
        user_id, item_id, rating, timestamp = line.strip().split(",")
        rating = float(rating)
        timestamp = int(timestamp)
        result.append(Action(user_id, item_id, timestamp, {"rating": rating}))
    return result