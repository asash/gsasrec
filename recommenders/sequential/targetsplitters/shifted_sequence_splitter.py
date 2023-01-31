from random import Random
from aprec.recommenders.sequential.targetsplitters.targetsplitter import TargetSplitter


class ShiftedSequenceSplitter(TargetSplitter):
    def __init__(self) -> None:
        super().__init__()
    
    def split(self, sequence):
        train = sequence[-self.seqence_len - 1: -1]
        label = sequence[-len(train):]
        return train, label
    
