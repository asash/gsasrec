import copy
from aprec.recommenders.sequential.targetsplitters.targetsplitter import TargetSplitter


class SequenceContinuation(TargetSplitter):
    def __init__(self, add_cls=False) -> None:
        super().__init__()
        self.add_cls = add_cls
    
    def split(self, sequence, max_targets=1):
        if len(sequence) == 0:
            return [], []
        train = sequence[:-max_targets]
        
        target = sequence[-max_targets:]

        if self.add_cls:
            cls_token = self.num_items + 1 #self.num_items is used for padding
            for t in target:
                cls = (t[0], cls_token)
                train.append(cls)
        return train, target