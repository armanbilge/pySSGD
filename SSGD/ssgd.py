import sys
import itertools as it
import numpy as np
np.seterr('ignore')

from scipy.optimize import minimize

from SSGD.Integrator import Integrator
from SSGD.PairedCompositeLikelihood import PairedCompositeLikelihood
from SSGD.SequenceErrorModel import SequenceErrorModel
from SSGD.SiteModel import SiteModel
from SSGD.Taxon import Taxon

perfectTaxaFile = sys.argv[1]
erroredTaxaFile = sys.argv[2]
patternsFile = sys.argv[3]

def readTaxa(fn):
    with open(fn) as f:
        return [Taxon(a, float(b)) for a, b in map(str.split, filter(lambda l: not l.isspace(), f))]

perfectTaxa = readTaxa(perfectTaxaFile)
erroredTaxa = readTaxa(erroredTaxaFile)
taxa = perfectTaxa + erroredTaxa

with open(patternsFile) as f:
    patterns = list(map(eval, filter(lambda l: not l.isspace(), f)))

def getFrequencies(patterns):
    pattern = next(iter(patterns.values()))
    freq = np.array([pattern.get((i, i), 0) for i in range(4)])
    return freq / sum(freq)

errorModel = SequenceErrorModel(erroredTaxa)
# [10000, 10000, float('inf')]
likelihoods = [PairedCompositeLikelihood(taxa, pattern, Integrator([0.25] * 4, [0, float('inf')]), SiteModel(1), errorModel) for pattern in patterns]

N = len(erroredTaxa) + len(likelihoods) * 3

scale = np.array([0] * len(erroredTaxa) + [1E-9, 1E6, 1E6] * len(likelihoods))

def f(args):
    print(args)
    args *= scale
    print(args)
    errorRate = args[:len(erroredTaxa)]
    errorModel.setErrorRate(errorRate)
    for likelihood, parameters in zip(likelihoods, np.split(args[len(erroredTaxa):], len(likelihoods))):
        c = it.count()
        likelihood.siteModel.setMu(parameters[next(c)])
        # likelihood.siteModel.setAlpha(parameters[next(c)])
        likelihood.siteModel.setAlpha(1.0)
        # likelihood.integrator.setKappa(parameters[next(c)])
        likelihood.integrator.setKappa(1.0)
        likelihood.integrator.setTheta(parameters[next(c):])
    l = - sum(map(PairedCompositeLikelihood.logLikelihood, likelihoods))
    print(l)
    return l


R = minimize(f, [1] * N, method='L-BFGS-B', bounds=([(1.0E-2, 1.0E2), (1.0E-6, 1.0E4), (1.0E-6, 1.0E4)]), tol=1e-20, options={'disp': True, 'iprint': 2, 'ftol': 1, 'gtol': 1E-8})
print(R)