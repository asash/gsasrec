from aprec.evaluation.metrics.metric import Metric


class HighestScore(Metric):
    def __init__(self):
        self.name = "HighestScore"
        
    def __call__(self, recommendations, actual_actions):
        if len(recommendations) == 0:
            return 0
        return recommendations[0][1] 