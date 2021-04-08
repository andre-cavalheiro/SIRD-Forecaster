from os import getcwd
import sys
sys.path.append(getcwd() + '/..')
from os.path import join
import traceback
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import libs.pandasLib as pl
from utils.loader import pullNationalData
from utils.calculations import calculateNationalRecoveries, projectSIRDParamsForward, pseudoCountsUpdate
from sird.sirdRegionalWrapper import regionalWrapper
from sird.sirdNationalWrapper import nationalWrapper


def initSimulation(df, populationDf, targetRegions, sirdPeriod, applyConstantParams, constantParams, movAvgInterval):

    simulationOriginDay = sirdPeriod[0]

    # Init model wrapper
    wp = regionalWrapper(df.reset_index(), populationDf, targetRegions, simulationOriginDay, sirdPeriod, applyConstantParams, constantParams)

    # Apply moving average to time series
    maCols = ["new-cases", "new-recovered", "new-deaths", "total-cases", "total-recovered", "total-deaths", "total-active-cases"]
    wp.movingAvg(maCols, movAvgInterval)

    # Use historic data to calculate parameters
    wp.calculateSIRDParametersByDay()

    # Smooth parameter time series
    wp.smoothSIRDParameters()

    return wp


def setForecastingParams(wp, projectionMethod, forecastPeriod, projectionParams={}, pseudoCounts=False, nationalParams=None, populationDf=None):

    paramLabels = ['beta', 'gamma', 'delta']

    # Get historic parameter evolution (smoothed)
    paramsByRegion = wp.getColumnsByRegion(columns=paramLabels+['R0'])

    if pseudoCounts is True:
        assert(nationalParams is not None and populationDf is not None)
        paramsByRegion = pseudoCountsUpdate(paramsByRegion, nationalParams, populationDf, projectionParams['pseudoCountsK'], params=paramLabels+['R0'])

    # Project parameters to the future
    paramsByRegion = projectSIRDParamsForward(paramsByRegion, projectionMethod, paramLabels, len(forecastPeriod)+1, projectionParams)
    highestParamValue = 0.4  # fixme

    return paramsByRegion, highestParamValue


def forecast(wp, predictedParams, forecastPeriod):
    initialConditionsDate = forecastPeriod[0]-timedelta(days=1)
    # Make forecast
    resultsByRegion = wp.sirdSimulation(predictedParams, initialConditionsDate=initialConditionsDate)

    # Transform into dataframe
    resultsDf = wp.transformIntoDf(resultsByRegion, forecastPeriod)
    return resultsDf


def nationalSIRD(regions, historicPeriod, sirdPeriod, movAvgInterval):

    nationalDf, _ = pullNationalData(regions, historicPeriod)
    nationalDf = calculateNationalRecoveries(nationalDf, historicPeriod)
    nationalDf['Region'] = 'PORTUGAL'
    nationalDf = nationalDf.reset_index()
    nw = nationalWrapper(nationalDf, sirdPeriod[0], sirdPeriod)

    # Apply moving average to time series
    maCols = ["new-cases", "new-recovered", "new-deaths", "total-cases", "total-recovered", "total-deaths", "total-active-cases"]
    nw.movingAvg(maCols, movAvgInterval)

    # Use historic data to calculate parameters
    nw.calculateSIRDParametersByDay()

    # Smooth parameter time series
    nw.smoothSIRDParameters()

    nationalParams = nw.getParams(columns=["beta", "gamma", "delta", "R0"])

    return nationalParams




