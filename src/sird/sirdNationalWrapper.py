from os import getcwd
import sys
sys.path.append(getcwd() + '/..')
import logging
from datetime import date, timedelta

from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val
import pandas as pd
import numpy as np
import wandb

from sird.SIRD import SIRD
from sird.sirdRegionalWrapper import regionalWrapper, k_smooth


class nationalWrapper(regionalWrapper):

    def __init__(self, data, firstDate, dataTimePeriod):
        self.df = data
        self.initialDate = firstDate        # The first valid date in the dataset
        self.dfTimePeriod = dataTimePeriod[dataTimePeriod.index(firstDate):]
        assert(self.dfTimePeriod[0] == self.initialDate)
        self.preprocess()

    def preprocess(self):
        self.df.Day = pd.to_datetime(self.df.Day, format='%Y-%m-%d')

        # Filter by time period
        self.df = self.df.loc[self.df.Day >= self.initialDate]
        self.df = self.df.loc[self.df.Day <= self.dfTimePeriod[-1]]

        # Index by day
        self.df = self.df.set_index("Day", drop=True).sort_index()
        logging.info(f'Performing SIRD simulation for PORTUGAL')

    def movingAvg(self, columns, period):
        for column in columns:
            self.df[column] = self.df[column].rolling(period, min_periods=1).mean()
        return self.df

    def smoothSIRDParameters(self, bandwidth=None):

        if bandwidth is None:
            bandwidth = {"beta": None, "gamma": None, "delta": None}

        '''
        old = {}
        old['beta'] = self.df.loc['beta']
        old['gamma'] = self.df.loc['gamma']
        old['delta'] = self.df.loc['delta']
        '''

        self.df['beta'] = k_smooth(self.df['beta'].values, smoothing_parameter=bandwidth['beta'])

        self.df['gamma'] = k_smooth(self.df['gamma'].values, smoothing_parameter=bandwidth['gamma'])

        self.df['delta'] = k_smooth(self.df['delta'].values, smoothing_parameter=bandwidth['delta'])
        self.df['R0'] = k_smooth(self.df['R0'].values, smoothing_parameter=None)

        #################################################################
        # Sanity plots
        '''
        import matplotlib.pyplot as plt
        columns = ['beta', 'gamma', 'delta']

        for c in columns:
            ts = self.df.loc[c].values
            idx = [i for i in range(len(ts))]
            plt.plot(idx, ts, label=c)

            plt.title(f'{region} - AFTER smooth')
            plt.legend()
            plt.show()
            plt.close()

        for c in columns:
            idx = [i for i in range(len(old[region][c]))]
            plt.plot(idx, old[region][c], label=f'old-{c}')

        plt.title(f'{region} - BEFORE smooth')
        plt.legend()
        plt.show()
        plt.close()
        '''
        ##################################################################

        return self.df

    def calculateSIRDParametersByDay(self):
        """
        Based on this article https://www.medrxiv.org/content/10.1101/2020.05.28.20115527v1.full.pdf
        """
        self.df = self.df.assign(beta=self.df["new-cases"]/self.df["total-active-cases"]).fillna(0)\
            .replace([np.inf, -np.inf], 0)

        self.df = self.df.assign(gamma=(self.df["new-recovered"]/self.df["total-active-cases"]).fillna(0)
            .replace([np.inf, -np.inf],0))

        self.df = self.df.assign(delta=(self.df["new-deaths"]/self.df["total-active-cases"]).fillna(0)
            .replace([np.inf, -np.inf], 0))

        # R_0 goes here
        self.df = self.df.assign(R0=self.df["beta"] * 14)

        return self.df

    def calcInitialConditions(self, startingDate):

        n0 = 10280000
        susceptible = n0 - self.df.loc[startingDate, "total-active-cases"] - self.df.loc[
            startingDate, "total-recovered"] - self.df.loc[startingDate, "total-deaths"]
        active = self.df.loc[startingDate, "total-active-cases"]
        recovered = self.df.loc[startingDate, "total-recovered"]
        deaths = self.df.loc[startingDate, "total-deaths"]

        return (susceptible, active, recovered, deaths)

    def sirdSimulation(self, paramsByRegion, simulationStartDate=None):

        if simulationStartDate is None:
            simulationStartDate = self.initialDate

        beta, gamma, delta = paramsByRegion['beta'], paramsByRegion['gamma'], paramsByRegion['delta']

        simulationAxis = np.arange(len(beta))

        beta = interp1d(simulationAxis, beta, bounds_error=False, fill_value="extrapolate")
        gamma = interp1d(simulationAxis, gamma, bounds_error=False, fill_value="extrapolate")
        delta = interp1d(simulationAxis, delta, bounds_error=False, fill_value="extrapolate")

        initialConditions = self.calcInitialConditions(simulationStartDate)

        model = SIRD(N=initialConditions[0], beta=beta, gamma=gamma, delta=delta)

        linSpaceAxis = np.linspace(0, len(simulationAxis), len(simulationAxis))
        results = model.simulate(initialConditions, linSpaceAxis)

        return results

    def getParams(self, columns=["beta", "gamma", "delta"]):
        output = {'PORTUGAL': {}}
        for column in columns:
            output['PORTUGAL'][column] = self.df[column]
        return output