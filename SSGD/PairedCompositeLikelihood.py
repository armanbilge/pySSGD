import itertools as it
import numpy as np

class PairedCompositeLikelihood:

    def __init__(self, taxa, patterns, integrator, siteModel, errorModel):
        self.taxa = {taxon.id: taxon.height for taxon in taxa}
        self.patterns = patterns
        self.integrator = integrator
        self.siteModel = siteModel
        self.errorModel = errorModel
        self.scale = 0.0

    def logLikelihood(self):
        def f(a, b):
            g = lambda i, j: self.pairLogLikelihood(a, self.errorModel.getPartial(a, i), b, self.errorModel.getPartial(b, j))
            x = list(w * g(i, j) for ((i, j), w) in self.patterns[tuple(sorted((a, b)))].items())
            return sum(x)
        return sum(f(a, b) for a, b in it.combinations(self.taxa, 2))

    def pairLogLikelihood(self, a, aPartial, b, bPartial):
        mu, prop = self.siteModel.getCategories()
        f = lambda i, j: sum(self.integrator.freq[i] * aPartial[i] * bPartial[j] * self.integrator.integratedProbability(i, self.taxa[a], j, self.taxa[b], mu) * prop)
        return np.log(sum(f(i, j) for i, j in it.product(range(4), repeat=2)))
