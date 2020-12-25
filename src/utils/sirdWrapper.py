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

from utils.SIRD import SIRD


def k_smooth(y, smoothing_parameter=None):
    nw = ks.NadarayaWatsonSmoother(smoothing_parameter=smoothing_parameter)
    y = skfda.representation.grid.FDataGrid(y)
    nw.fit(y)
    nw_y = nw.transform(y)
    return nw_y.data_matrix.reshape(-1)

class Wrapper:

    def __init__(self, data, populationDf, regions, firstDate, dataTimePeriod, applyconstantParams=False, constantParams={}):
        self.data = data
        self.regions = regions
        self.populationDf = populationDf
        self.initialDate = firstDate        # The first valid date in the dataset
        self.dataTimePeriod = dataTimePeriod[dataTimePeriod.index(firstDate):]
        assert(self.dataTimePeriod[0] == self.initialDate)

        self.applyconstantParams = applyconstantParams
        if applyconstantParams is True:
            assert('beta' in constantParams.keys() and 'gamma' in constantParams.keys() and 'delta' in constantParams.keys())
            self.constantParams = constantParams

        self.preprocess()

    def preprocess(self):
        self.data.Day = pd.to_datetime(self.data.Day, format='%Y-%m-%d')

        # Filter by time period
        self.data = self.data.loc[self.data.Day >= self.initialDate]
        self.data = self.data.loc[self.data.Day <= self.dataTimePeriod[-1]]

        # Keep only relevant regions
        self.data = self.data[self.data['Region'].isin(self.regions+['PORTUGAL'])]

        # Index by day
        self.data = self.data.set_index("Day", drop=True).sort_index()

        logging.info(f'Performing SIRD simulation for {len(self.regions)} regions')

    def movingAvg(self, columns, period):

        for region in self.regions:
            for column in columns:
                placeholder = self.data[self.data.Region == region][column].values

                self.data.loc[self.data.Region==region, column] = self.data[self.data.Region==region][column]\
                    .rolling(period, min_periods=1).mean()
                r=1
                #################################################################
                '''
                # Sanity plots
                import matplotlib.pyplot as plt
                new = self.data[self.data.Region == region][column].values
                idx = [i for i in range(len(new))]
                plt.plot(idx, placeholder, label='Orig')
                plt.plot(idx, new, label='MA')
                plt.title(f'{region}-{column}')

                # plt.show()
                plt.savefig(f'D:\Downloads\MA\{region}-{column}.png')
                plt.close()
                '''
                ##################################################################
        return self.data

    def smoothSIRDParameters(self, bandwidth=None):

        if bandwidth is None:
            bandwidth = {"beta": None, "gamma": None, "delta": None}

        old = {}
        for region in self.regions:
            old[region] = {}
            old[region]['beta'] = self.data.loc[self.data.Region == region, 'beta']
            old[region]['gamma'] = self.data.loc[self.data.Region == region, 'gamma']
            old[region]['delta'] = self.data.loc[self.data.Region == region, 'delta']

            self.data.loc[self.data.Region == region, 'beta'] = k_smooth(
                self.data.loc[self.data.Region == region, 'beta'].values, smoothing_parameter=bandwidth['beta'])

            self.data.loc[self.data.Region == region, 'gamma'] = k_smooth(
                self.data.loc[self.data.Region == region, 'gamma'].values, smoothing_parameter=bandwidth['gamma'])

            self.data.loc[self.data.Region == region, 'delta'] = k_smooth(
                self.data.loc[self.data.Region == region, 'delta'].values, smoothing_parameter=bandwidth['delta'])

        #################################################################
        # Sanity plots
        '''
        import matplotlib.pyplot as plt
        columns = ['beta', 'gamma', 'delta']
        for region in self.regions:

            for c in columns:
                ts = self.data.loc[self.data.Region == region, c].values
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

            r=1
        '''
        ##################################################################

        return self.data

    def calculateSIRDParametersByDay(self):
        """
        Based on this article https://www.medrxiv.org/content/10.1101/2020.05.28.20115527v1.full.pdf
        """

        self.data = self.data.assign(
            beta=self.data["new-cases"] / self.data["total-active-cases"]).fillna(0).replace([np.inf, -np.inf], 0)

        self.data = self.data.assign(
            gamma=(self.data["new-recovered"] / self.data["total-active-cases"]).fillna(0).replace([np.inf, -np.inf],
                                                                                                   0))
        self.data = self.data.assign(
            delta=(self.data["new-deaths"] / self.data["total-active-cases"]).fillna(0).replace([np.inf, -np.inf], 0))

        if self.applyconstantParams:
            logging.info('!!! Setting some parameters to const !!!')
            self.makeConstantCasesRegions()

        return self.data

    def makeConstantCasesRegions(self):
        def function(row):
            if row.loc['total-active-cases'] < 100:
                row.loc['gamma'] = self.constantParams['gamma']
                row.loc['beta'] = self.constantParams['gamma']
                row.loc['delta'] = self.constantParams['gamma']
            return row
        self.data = self.data.apply(function, axis=1)

    def calcInitialConditions(self, region, startingDate):

        specificDf = self.data.loc[self.data.Region == region]
        n0 = self.populationDf[region]

        susceptible = n0 - specificDf.loc[startingDate, "total-active-cases"] - specificDf.loc[
                    startingDate, "total-recovered"] - specificDf.loc[startingDate, "total-deaths"]
        active = specificDf.loc[startingDate, "total-active-cases"]
        recovered = specificDf.loc[startingDate, "total-recovered"]
        deaths = specificDf.loc[startingDate, "total-deaths"]

        return (susceptible, active, recovered, deaths)

    def sirdSimulation(self, paramsByRegion, simulationStartDate=None):

        if simulationStartDate is None:
            simulationStartDate = self.initialDate

        results = {}
        for region in self.regions:

            beta, gamma, delta = paramsByRegion[region]['beta'], paramsByRegion[region]['gamma'], paramsByRegion[region]['delta']

            simulationAxis = np.arange(len(beta))
            beta = interp1d(simulationAxis, beta, bounds_error=False, fill_value="extrapolate")
            gamma = interp1d(simulationAxis, gamma, bounds_error=False, fill_value="extrapolate")
            delta = interp1d(simulationAxis, delta, bounds_error=False, fill_value="extrapolate")

            initialConditions = self.calcInitialConditions(region, simulationStartDate)
            model = SIRD(
                N=initialConditions[0],
                beta=beta,
                gamma=gamma,
                delta=delta,
            )

            linSpaceAxis = np.linspace(0, len(simulationAxis), len(simulationAxis))
            results[region] = model.simulate(initialConditions, linSpaceAxis)

            # results[region]['initialTotalCases'] = self.data.loc[self.data.Region == region].at[lastRealDataDate, 'total-cases']

            logging.info(f'Simulated {region}')

        return results

    def getParamsByRegion(self):
        return self.getColumnsByRegion(["beta", "gamma", "delta"])

    def getColumnsByRegion(self, columns):
        output = {r: {} for r in self.regions}
        for region in self.regions:
            specificDf = self.data.loc[self.data.Region == region]
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
                r=1

        return outputDf



class nationalWrapper:

    def __init__(self, data, firstDate, dataTimePeriod):
        self.data = data
        self.initialDate = firstDate        # The first valid date in the dataset
        self.dataTimePeriod = dataTimePeriod[dataTimePeriod.index(firstDate):]
        assert(self.dataTimePeriod[0] == self.initialDate)
        self.preprocess()

    def preprocess(self):
        self.data.Day = pd.to_datetime(self.data.Day, format='%Y-%m-%d')

        # Filter by time period
        self.data = self.data.loc[self.data.Day >= self.initialDate]
        self.data = self.data.loc[self.data.Day <= self.dataTimePeriod[-1]]

        # Index by day
        self.data = self.data.set_index("Day", drop=True).sort_index()
        logging.info(f'Performing SIRD simulation for PORTUGAL')

    def movingAvg(self, columns, period):
        for column in columns:
            self.data[column] = self.data[column].rolling(period, min_periods=1).mean()
        return self.data

    def smoothSIRDParameters(self, bandwidth=None):

        if bandwidth is None:
            bandwidth = {"beta": None, "gamma": None, "delta": None}

        '''
        old = {}
        old['beta'] = self.data.loc['beta']
        old['gamma'] = self.data.loc['gamma']
        old['delta'] = self.data.loc['delta']'''

        self.data['beta'] = k_smooth(self.data['beta'].values, smoothing_parameter=bandwidth['beta'])

        self.data['gamma'] = k_smooth(self.data['gamma'].values, smoothing_parameter=bandwidth['gamma'])

        self.data['delta'] = k_smooth(self.data['delta'].values, smoothing_parameter=bandwidth['delta'])

        #################################################################
        # Sanity plots
        '''
        import matplotlib.pyplot as plt
        columns = ['beta', 'gamma', 'delta']

        for c in columns:
            ts = self.data.loc[c].values
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

        return self.data

    def calculateSIRDParametersByDay(self):
        """
        Based on this article https://www.medrxiv.org/content/10.1101/2020.05.28.20115527v1.full.pdf
        """
        self.data = self.data.assign(beta=self.data["new-cases"]/self.data["total-active-cases"]).fillna(0)\
            .replace([np.inf, -np.inf], 0)

        self.data = self.data.assign(gamma=(self.data["new-recovered"]/self.data["total-active-cases"]).fillna(0)
            .replace([np.inf, -np.inf],0))

        self.data = self.data.assign(delta=(self.data["new-deaths"]/self.data["total-active-cases"]).fillna(0)
            .replace([np.inf, -np.inf], 0))

        return self.data

    def calcInitialConditions(self, startingDate):

        n0 = 10280000
        susceptible = n0 - self.data.loc[startingDate, "total-active-cases"] - self.data.loc[
            startingDate, "total-recovered"] - self.data.loc[startingDate, "total-deaths"]
        active = self.data.loc[startingDate, "total-active-cases"]
        recovered = self.data.loc[startingDate, "total-recovered"]
        deaths = self.data.loc[startingDate, "total-deaths"]

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

    def getParams(self):
        output = {'PORTUGAL': {}}
        for column in ["beta", "gamma", "delta"]:
            output['PORTUGAL'][column] = self.data[column]
        return output