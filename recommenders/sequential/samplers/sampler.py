from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters


class NegativesSampler(object):
    def __init__(self, data_parameters: SequentialDataParameters, num_negatives: int) -> None:
        self.data_parameters = data_parameters
        self.num_negatives = num_negatives

    def fit(self, train_users):
        pass

    def __call__(self, masked_sequences, labels):
        raise NotImplementedError()


def get_negatives_sampler(sampler_name, data_parameters, num_negatives) -> NegativesSampler:
    if sampler_name == "random":
        from aprec.recommenders.sequential.samplers.random_sampler import RandomNegativesSampler
        return RandomNegativesSampler(data_parameters, num_negatives)

    elif sampler_name == "popularity":
        from aprec.recommenders.sequential.samplers.popularity_sampler import PopularitySampler
        return PopularitySampler(data_parameters, num_negatives)

    elif sampler_name == "idf":
        from aprec.recommenders.sequential.samplers.idf_sampler import IDFSampler
        return IDFSampler(data_parameters, num_negatives)
    else:
        raise Exception(f"wrong negatives sampler name {sampler_name}")  