class Metric(object):
    less_is_better = False
    def __init__(self):
        self.name == "undefined"
    
    def get_name(self) -> str:
        return self.name

    def __call__(self, recommendations, actual):
        raise NotImplementedError
