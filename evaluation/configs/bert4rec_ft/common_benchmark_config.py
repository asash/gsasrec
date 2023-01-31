from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsWithReplacementSampler
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'),
                     Confidence('Sigmoid'),
                     Entropy('Softmax', 10), 
                     HIT(10)]

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200
EMBEDDING_SIZE=128
 
def vanilla_sasrec(loss='bce', num_samples=1, batch_size=128):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=EMBEDDING_SIZE, loss=loss, vanilla_num_negatives=num_samples)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=batch_size)

def gSASRec(num_samples=32, t=0.5):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(vanilla=True, embedding_size=EMBEDDING_SIZE, loss='bce', vanilla_num_negatives=num_samples, 
                                vanilla_bce_t=t)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=128)


def sasrec_full_target():
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    model_config = SASRecConfig(full_target=True, loss='softmax_ce', embedding_size=EMBEDDING_SIZE)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=64)


def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=128,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                eval_batch_size=256, #no need for gradients, should work ok
                                validation_batch_size=256,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

def get_bert_style_model(model_config, tuning_samples_portion, batch_size=128):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        recommender_config = SequentialRecommenderConfig(model_config, 
                                               train_epochs=10000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               eval_batch_size=256, #no need for gradients, should work ok
                                               validation_batch_size=256,
                                               sequence_splitter=lambda: ItemsMasking(tuning_samples_prob=tuning_samples_portion), 
                                               max_batches_per_epoch=batch_size,
                                               targets_builder=ItemsMaskingTargetsBuilder,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               extra_val_metrics = EXTRA_VAL_METRICS)
        
        return SequentialRecommender(recommender_config)

def full_bert(loss='softmax_ce', tuning_samples_portion=0.0):
        from aprec.recommenders.sequential.models.bert4rec.full_bert import FullBERTConfig
        model_config =  FullBERTConfig(embedding_size=EMBEDDING_SIZE, loss=loss)
        return get_bert_style_model(model_config, tuning_samples_portion=tuning_samples_portion, batch_size=64)

def sampling_bert(sampling_strategy, num_samples, loss):
        from aprec.recommenders.sequential.models.bert4rec.bert4recft import SampleBERTConfig
        model_config =  SampleBERTConfig(embedding_size=EMBEDDING_SIZE, loss=loss, num_negative_samples=num_samples, sampler=sampling_strategy)
        return get_bert_style_model(model_config, 0.0, batch_size=64)

def popularity():
        return TopRecommender()

def mf_bpr():
        return LightFMRecommender(num_latent_components=EMBEDDING_SIZE, num_threads=32)


recommenders = {
        "popularity": popularity,
        "mf-bpr": mf_bpr
        }

recommenders = {}

recommenders["BERT4rec"] = full_bert
recommenders["SASRec-FullCE"] =  sasrec_full_target 
recommenders["gSASRec-samples:128-t:0.9"] = lambda: gSASRec(num_samples=128, t=0.9) 
recommenders["gSASRec"] =  gSASRec 
recommenders["gSASRec-t:0.9"] = lambda: gSASRec(t=0.9) 
recommenders["gSASRec-samples:128"] = lambda: gSASRec(num_samples=128) 
recommenders["SASRec-vanilla"] =  vanilla_sasrec 
recommenders["BERT4rec-negPerPos"] = lambda: sampling_bert(sampling_strategy='random', loss='bce', num_samples=1)
recommenders["popularity"] = popularity
recommenders["mf_bpr"] = mf_bpr


for num_samples in [1, 4, 16, 64, 256]:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                recommenders[f"DebSASRec-samples:{num_samples}-t:{t}"] = lambda num_samples=num_samples, t=t: gSASRec(num_samples=num_samples, t=t)

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

