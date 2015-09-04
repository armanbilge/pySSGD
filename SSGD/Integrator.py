import numpy as np

class Integrator:

    def __init__(self, freq, epochDuration):
        self.cache = {}
        self.freq = freq
        self.kappa = 1
        self.betaKnown = False
        self.epochDuration = np.array(epochDuration)
        self.theta = np.array([1] * (len(epochDuration) + 1))

    def setKappa(self, kappa):
        self.kappa = kappa
        self.betaKnown = False
        self.cache.clear()

    def setTheta(self, theta):
        self.theta = np.array(theta)
        self.cache.clear()

    def integratedProbability(self, iState, iTime, jState, jTime, mu):
        if iTime > jTime:
            return self.integratedProbability(jState, jTime, iState, iTime, mu)
        x = (iState, iTime, jState, jTime, tuple(mu))
        if x not in self.cache:
            self.cache[x] = self.calculateIntegratedProbability(iState, iTime, jState, jTime, mu)
        return self.cache[x]

    def calculateIntegratedProbability(self, iState, iTime, jState, jTime, mu):

        if not self.betaKnown:
            self.calculateBeta()

        tau = np.abs(iTime - jTime)

        if iState % 2 == jState % 2: # transition
            H = lambda t, N: self.transitionH(t, N, iState, jState, tau, mu)
        else: # transversion
            H = lambda t, N: self.transversionH(t, N, jState, tau, mu)

        return self.integrateIntervals(H, max(iTime, jTime))

    def integrateIntervals(self, H, start):

        m = len(self.epochDuration)

        k = 0
        current = 0
        while current <= start:
            current += self.epochDuration[k]
            k += 1
        previous = start

        g = 1.0
        integratedP = 0
        for i in range(k, m):

            N = self.theta[i - 1]
            integratedP += g * np.exp(previous / N) * (H(current, N) - H(previous, N))

            g *= np.exp(-(current - previous) / N)

            previous = current
            current += self.epochDuration[i]

        N = self.theta[m - 1]
        integratedP -= g * np.exp(previous / N) * H(previous, N)

        return np.nan_to_num(integratedP)

    def calculateBeta(self):
        kappa = self.kappa
        freqA, freqC, freqG, freqT = self.freq
        freqR = freqA + freqG
        freqY = freqC + freqT
        self.beta = 1 / (2 * (freqR * freqY + kappa * (freqA * freqG + freqC * freqT)))
        self.betaKnown = True

    def transitionH(self, t, N, i, j, tau, mu):

        ihat = (i + 2) % 4
        pm = -2 * int(i == j) + 1

        betamu = self.beta * mu
        twobetamuN = 2 * betamu * N

        mbetamutwotptau = -betamu * (2*t + tau)

        freqi = self.freq[i]
        freqj = self.freq[j]
        freqihat = self.freq[ihat]
        freq = freqi + freqihat

        freqkappam1p1 = freq * (self.kappa - 1) + 1

        return np.exp(-t/N) * (pm * freqihat * np.exp(mbetamutwotptau * freqkappam1p1) / (twobetamuN * freqkappam1p1 + 1) - freqj * ((1 - freq) * np.exp(mbetamutwotptau) / (twobetamuN + 1) + freq)) / freq

    def transversionH(self, t, N, j, tau, mu):
        betamu = self.beta * mu
        return self.freq[j] * np.exp(-t/N) * (np.exp(-betamu * (2*t + tau)) / (2 * betamu * N + 1) - 1)