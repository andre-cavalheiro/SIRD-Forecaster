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

from sird.SIRD import SIRD


def k_smooth(y, smoothing_parameter=None):
    nw = ks.NadarayaWatsonSmoother(smoothing_parameter=smoothing_parameter)
    y = skfda.representation.grid.FDataGrid(y)
    nw.fit(y)
    nw_y = nw.transform(y)
    return nw_y.data_matrix.reshape(-1)

class regionalWrapper:

    def __init__(self, data, populationDf, regions, firstDate, dataTimePeriod, applyconstantParams=False, constantParams={}):
        self.df = data
        self.regions = regions
        self.populationDf = populationDf
        self.initialDate = firstDate        # The first valid date in the dataset
        self.dfTimePeriod = dataTimePeriod[dataTimePeriod.index(firstDate):]
        assert(self.dfTimePeriod[0] == self.initialDate)

        self.applyconstantParams = applyconstantParams
        if applyconstantParams is True:
            assert('beta' in constantParams.keys() and 'gamma' in constantParams.keys() and 'delta' in constantParams.keys())
            self.constantParams = constantParams

        self.preprocess()

    def preprocess(self):
        self.df.Day = pd.to_datetime(self.df.Day, format='%Y-%m-%d')

        # Filter by time period
        self.df = self.df.loc[self.df.Day >= self.initialDate]
        self.df = self.df.loc[self.df.Day <= self.dfTimePeriod[-1]]

        # Keep only relevant regions
        self.df = self.df[self.df['Region'].isin(self.regions+['PORTUGAL'])]

        # Index by day
        self.df = self.df.set_index("Day", drop=True).sort_index()

        logging.info(f'Performing SIRD simulation for {len(self.regions)} regions')

    def movingAvg(self, columns, period):
        for region in self.regions:
            for column in columns:
                self.df.loc[self.df.Region==region, column] = self.df[self.df.Region==region][column]\
                    .rolling(period, min_periods=1).mean()
        return self.df.reset_index()

    def calculateSIRDParametersByDay(self):
        """
        Based on this article https://www.medrxiv.org/content/10.1101/2020.05.28.20115527v1.full.pdf
        """

        self.df = self.df.assign(
            beta=self.df["new-cases"] / self.df["total-active-cases"]).fillna(0).replace([np.inf, -np.inf], 0)

        self.df = self.df.assign(
            gamma=(self.df["new-recovered"] / self.df["total-active-cases"]).fillna(0).replace([np.inf, -np.inf],
                                                                                                   0))
        self.df = self.df.assign(
            delta=(self.df["new-deaths"] / self.df["total-active-cases"]).fillna(0).replace([np.inf, -np.inf], 0))

        if self.applyconstantParams:
            logging.info('!!! Setting some parameters to const !!!')
            self.applyConstantParams()

        # R_0 goes here
        # R_0 = beta * (1/lamda)
        self.df = self.df.assign(R0=self.df["beta"] * 14)

        return self.df

    def smoothSIRDParameters(self, bandwidth=None):

        if bandwidth is None:
            bandwidth = {"beta": None, "gamma": None, "delta": None}

        old = {}
        for region in self.regions:
            old[region] = {}
            old[region]['beta'] = self.df.loc[self.df.Region == region, 'beta']
            old[region]['gamma'] = self.df.loc[self.df.Region == region, 'gamma']
            old[region]['delta'] = self.df.loc[self.df.Region == region, 'delta']
            old[region]['R0'] = self.df.loc[self.df.Region == region, 'R0']


            self.df.loc[self.df.Region == region, 'beta'] = k_smooth(
                self.df.loc[self.df.Region == region, 'beta'].values, smoothing_parameter=bandwidth['beta'])

            self.df.loc[self.df.Region == region, 'gamma'] = k_smooth(
                self.df.loc[self.df.Region == region, 'gamma'].values, smoothing_parameter=bandwidth['gamma'])

            self.df.loc[self.df.Region == region, 'delta'] = k_smooth(
                self.df.loc[self.df.Region == region, 'delta'].values, smoothing_parameter=bandwidth['delta'])

            self.df.loc[self.df.Region == region, 'R0'] = k_smooth(
                self.df.loc[self.df.Region == region, 'R0'].values, smoothing_parameter=None)

        #################################################################
        '''
        # Sanity plots
        import matplotlib.pyplot as plt
        columns = ['beta', 'gamma', 'delta']
        for region in self.regions:

            for c in columns:
                ts = self.df.loc[self.df.Region == region, c].values
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

    def applyConstantParams(self, threshold=100):
        def function(row):
            if row.loc['total-active-cases'] < threshold:
                row.loc['gamma'] = self.constantParams['gamma']
                row.loc['beta'] = self.constantParams['beta']
                row.loc['delta'] = self.constantParams['delta']
            return row
        self.df = self.df.apply(function, axis=1)

    def calcInitialConditions(self, region, startingDate):

        specificDf = self.df.loc[self.df.Region == region]
        n0 = self.populationDf[region]

        susceptible = n0 - specificDf.loc[startingDate, "total-active-cases"] - specificDf.loc[
                    startingDate, "total-recovered"] - specificDf.loc[startingDate, "total-deaths"]
        active = specificDf.loc[startingDate, "total-active-cases"]
        recovered = specificDf.loc[startingDate, "total-recovered"]
        deaths = specificDf.loc[startingDate, "total-deaths"]

        return (susceptible, active, recovered, deaths)

    def sirdSimulation(self, paramsByRegion, initialConditionsDate=None):

        if initialConditionsDate is None:
            initialConditionsDate = self.initialDate

        results = {}
        for region in self.regions:

            beta, gamma, delta = paramsByRegion[region]['beta'], paramsByRegion[region]['gamma'], paramsByRegion[region]['delta']

            simulationAxis = np.arange(len(beta))
            beta = interp1d(simulationAxis, beta, bounds_error=False, fill_value="extrapolate")
            gamma = interp1d(simulationAxis, gamma, bounds_error=False, fill_value="extrapolate")
            delta = interp1d(simulationAxis, delta, bounds_error=False, fill_value="extrapolate")

            initialConditions = self.calcInitialConditions(region, initialConditionsDate)
            model = SIRD(
                N=initialConditions[0],
                beta=beta,
                gamma=gamma,
                delta=delta,
            )

            linSpaceAxis = np.linspace(0, len(simulationAxis), len(simulationAxis))
            results[region] = model.simulate(initialConditions, linSpaceAxis)

            # results[region]['initialTotalCases'] = self.df.loc[self.df.Region == region].at[lastRealDataDate, 'total-cases']

            # logging.info(f'Simulated {region}')

        return results

    def getColumnsByRegion(self, columns=["beta", "gamma", "delta"]):
        output = {r: {} for r in self.regions}
        for region in self.regions:
            specificDf = self.df.loc[self.df.Region == region]
            for column in columns:
                output[region][column] = specificDf[column]
        return output

    def transformIntoDf(self, resultsByRegion, forecastPeriod):

        # First day of forecasting and last known day are the same.
        idx = pd.MultiIndex.from_product(
            [forecastPeriod, self.regions],
            names=['Day', 'Region']
        )
        outputDf = pd.DataFrame(np.nan, idx, ['predictedCases'])
        outputDf = outputDf.sort_index()
        outputDf['inputDay'] = date.today()

        for region, data in resultsByRegion.items():
            forecastCumulative = data['total_cases_sird']
            forecastNewCases = np.diff(forecastCumulative)      # This is why simulation length = forecast length + 1 -> because we 'loose' a value here
            assert(len(forecastNewCases) == len(forecastPeriod))

            for day, newCases in zip(forecastPeriod, forecastNewCases):
                outputDf.loc[(day, region), 'predictedCases'] = newCases

        return outputDf