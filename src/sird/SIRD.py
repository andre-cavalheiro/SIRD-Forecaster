import numpy as np
import pandas as pd
from scipy.integrate import odeint


class SIRD:
    
    def __init__(self, N, beta, gamma, delta):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _constant_params(self):
        return not (
                (isinstance(self.beta, (list, pd.core.series.Series, np.ndarray)))
            and (isinstance(self.gamma, (list, pd.core.series.Series, np.ndarray)))
            and (isinstance(self.delta, (list, pd.core.series.Series, np.ndarray)))
            )

    def _deriv(self, y, t):
        S, I, R, D = y
        dSdt = -self.beta(t) * S * I / self.N
        dIdt = self.beta(t) * S * I / self.N - self.gamma(t) * I - self.delta(t) * I
        dRdt = self.gamma(t) * I
        dDdt = self.delta(t) * I

        return dSdt, dIdt, dRdt, dDdt

    def simulate(self, y0, t):
        ret = odeint(self._deriv, y0, t)
        S, I, R, D = ret.T

        return {'S': S, 'I': I, 'R': R, 'D': D, 'total_cases_sird': I + R + D}

    def getParams(self):
        return {'beta': self.beta, 'gamma': self.gamma, 'delta': self.delta}