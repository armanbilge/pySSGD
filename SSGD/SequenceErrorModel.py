import numpy as np

class SequenceErrorModel:

    def __init__(self, taxa):
        self.taxa = {j:i for i, j in enumerate(taxa)}
        self.errorRate = np.array([0] * len(taxa))

    def setErrorRate(self, errorRate):
        self.errorRate = np.array(errorRate)

    def getPartial(self, taxon, i):
        p = self.errorRate[self.taxa[taxon]] if taxon in self.taxa else 0
        return np.array([1 - p if i == j else p/3 for j in range(4)])
