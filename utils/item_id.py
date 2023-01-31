from collections import Counter, defaultdict
from pathlib import PosixPath


class ItemId(object):
    def __init__(self):
        self.straight = {}
        self.reverse = {}
        self.get_count = Counter()

    def size(self):
        return len(self.straight)

    def get_id(self, item_id):
        if item_id not in self.straight:
            self.straight[item_id] = len(self.straight)
            self.reverse[self.straight[item_id]] = item_id
        self.get_count[item_id] += 1
        return self.straight[item_id]

    def has_id(self, id):
        return id in self.reverse

    def has_item(self, item_id):
        return item_id in self.straight

    def reverse_id(self, id):
        return self.reverse[id] 

    def save(self, file_name):
        with open(file_name, "w") as output:
            for item in self.straight:
                output.write(f"{item} {self.straight[item]}\n")
    
    @staticmethod
    def load(file_name):
        straight, reverse = {}, {}
        for line in open(file_name):
            external, internal = line.rstrip().split(" ")
            internal = int(internal)
            straight[external] = internal 
            reverse[internal] = external
        result = ItemId()
        result.straight = straight
        result.reverse = reverse
        return result