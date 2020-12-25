from os import getcwd
import sys
sys.path.append(getcwd() + '/..')
from os.path import join
import traceback
import os
import logging
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

import libs.pandasLib as pl
from utils.loader import pullNationalData
from utils.sirdWrapper import Wrapper, nationalWrapper
from utils.calculations import calculateNationalRecoveries

def projectParamsForwards(paramsByRegion, method, params, forecastLength, avgInterval):
    maxValue = 0

    for region, payload in paramsByRegion.items():

        if method == 'last':
            for p in params:
                paramsByRegion[region][f'{p}-forecast'] = [payload[p].tolist()[-1] for _ in range(forecastLength)]
        elif method == 'avg':
            assert(avgInterval is not None)
            for p in params:
                paramsByRegion[region][f'{p}-forecast'] = [np.mean(payload[p].tolist()[-avgInterval]) for _ in range(forecastLength)]
        elif method == 'extrapolation':
            from scipy import interpolate
            for p in params:
                y = payload[p].tolist()
                x = [i for i in range(len(y))]
                f = interpolate.interp1d(x, y, fill_value="extrapolate")
                paramsByRegion[region][f'{p}-forecast'] = [f(i + len(y)).tolist() for i in range(forecastLength)]
        else:
            raise Exception(f'Unknown parameter projection method - {method}')

        for p in params:
            if max(paramsByRegion[region][f'{p}-forecast']) > maxValue:
                maxValue = max(paramsByRegion[region][f'{p}-forecast'])
            if max(payload[p]) > maxValue:
                maxValue = max(payload[p])

    return paramsByRegion, maxValue


def pseudoCountsUpdate(paramsByRegion, projectedNationalParams, populationDf, params=['beta', 'gamma', 'delta'], k=1000):
    for r, dt in paramsByRegion.items():
        for p in params:
            numA = dt[p].multiply(k)
            numB = projectedNationalParams['PORTUGAL'][p].multiply(populationDf[r])
            denom = k+populationDf[r]
            paramsByRegion[r][p] = (numA+numB)/(denom)
    return paramsByRegion

def sirdForecast(df, populationDf, targetRegions, sirdPeriod, forecastPeriod, projectionMethod, movAvgInterval,
                 applyConstantParams, pseudoCounts, constantParams={}, paramPredictInterval=None, nationalParams=None):

    simulationOriginDay, forecastStartDate = sirdPeriod[0], sirdPeriod[-1]

    dataset = Wrapper(df.reset_index(), populationDf, targetRegions, simulationOriginDay, sirdPeriod, applyConstantParams, constantParams)

    # Apply moving average to time series
    dataset.movingAvg(["new-cases", "new-recovered", "new-deaths", "total-cases", "total-recovered", "total-deaths",
                       "total-active-cases"], movAvgInterval)

    # Use historic data to calculate parameters
    dataset.calculateSIRDParametersByDay()


    # Smooth parameter time series
    dataset.smoothSIRDParameters()

    # Get historic parameter evolution (smoothed)
    paramsByRegion = dataset.getParamsByRegion()

    # Project parameters to the future
    paramsByRegion, highestParam = projectParamsForwards(paramsByRegion, projectionMethod, ['beta', 'gamma', 'delta'],
                                                         len(forecastPeriod)+1, paramPredictInterval)
    if pseudoCounts is True:
        projectedNationalParams, _ = projectParamsForwards(nationalParams, projectionMethod, ['beta', 'gamma', 'delta'],
                                                         len(forecastPeriod)+1, paramPredictInterval)
        paramsByRegion = pseudoCountsUpdate(paramsByRegion, projectedNationalParams, populationDf)

    predictedParams = {
        r: {c: v[f'{c}-forecast'] for c in ['beta', 'gamma', 'delta']}
        for r, v in paramsByRegion.items()}

    # Make forecast
    resultsByRegion = dataset.sirdSimulation(predictedParams, simulationStartDate=forecastStartDate)

    # Transform into dataframe
    resultsDf = dataset.transformIntoDf(resultsByRegion, forecastPeriod)

    return resultsDf, paramsByRegion, highestParam






