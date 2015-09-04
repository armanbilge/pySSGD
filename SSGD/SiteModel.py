import numpy as np
from scipy.stats import gamma

class SiteModel:

    def __init__(self, categoryCount):
        self.mu = 1.0
        self.alpha = 1.0
        self.categoryCount = categoryCount
        self.props = np.array([1 / categoryCount] * categoryCount)

    def setMu(self, mu):
        self.mu = mu

    def setAlpha(self, alpha):
        self.alpha = alpha

    def getCategories(self):

        rates = np.array([self.mu])
        # rates = self.mu * gamma.ppf(np.array([(2*i + 1) / (2 * self.categoryCount) for i in range(self.categoryCount)]), self.alpha, scale=1/self.alpha)
        return rates, self.props