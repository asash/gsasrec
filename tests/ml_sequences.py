def ml_sequences(n_actions):
    from aprec.utils.generator_limit import generator_limit
    from aprec.datasets.movielens20m import get_movielens20m_actions
    from aprec.utils.item_id import ItemId
    from collections import defaultdict
    sequences_dict = defaultdict(list)
    actions = [action for action in generator_limit(get_movielens20m_actions(), n_actions)]
    actions.sort(key = lambda action: action.timestamp)
    item_ids = ItemId()
    for action in actions:
        sequences_dict[action.user_id].append((action.timestamp, item_ids.get_id(action.item_id)))
    sequences = list(sequences_dict.values())
    return sequences, item_ids


