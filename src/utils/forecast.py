from os import getcwd
import sys
sys.path.append(getcwd() + '/..')
from os.path import join
import traceback
from unidecode import unidecode
import os
import logging
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

import libs.pandasLib as pl
import libs.visualization as vz
from utils.sirdWrapper import Wrapper


def projectParamsForwards(paramsByRegion, method, params, forecastLength, avgInterval=4):
    futureParamsByRegion = {r: {} for r in paramsByRegion.keys()}
    for region, payload in paramsByRegion.items():
        if method == 'constant':
            for p in params:
                futureParamsByRegion[region][p] = [payload[p][-1] for _ in range(forecastLength)]
        elif method == 'avg':
            for p in params:
                futureParamsByRegion[region][p] = [np.mean(payload[p][-avgInterval]) for _ in range(forecastLength)]
        elif method == 'extrapolation':
            from scipy import interpolate
            for p in params:
                y = payload[p]
                x = [i for i in range(len(y))]
                f = interpolate.interp1d(x, y, fill_value="extrapolate")
                futureParamsByRegion[region][p] = [f(i + len(y)) for i in range(forecastLength)]
        else:
            raise Exception('Unknown parameter projection method')
    return futureParamsByRegion


def sirdForecast(df, populationDf, targetRegions, historicPeriod, simulationOriginDay, forecastStartDate, forecastLength,
                 projectionMethod, movAvgInterval=7, outputPlots=False, outputDir='.'):

    sirdPeriod = historicPeriod[historicPeriod.index(simulationOriginDay): historicPeriod.index(forecastStartDate)+1]
    dataset = Wrapper(df.reset_index(), populationDf, targetRegions, simulationOriginDay, sirdPeriod)

    # Apply moving average to time series
    dataset.movingAvg(["new-cases", "new-recovered", "new-deaths", "total-active-cases", "total-cases"], movAvgInterval)

    # Use historic data to calculate parameters
    dataset.calculateSIRDParametersByDay()

    # Smooth parameter time series
    dataset.smoothSIRDParameters()

    # Get historic parameter evolution (smoothed)
    paramsByRegion = dataset.getParamsByRegion()

    # Project parameters to the future
    simulationLength = forecastLength+1
    predictedParamsByRegion = projectParamsForwards(paramsByRegion, projectionMethod, ['beta', 'gamma', 'delta'], simulationLength)

    # Make forecast
    forecastPeriod = [forecastStartDate + timedelta(days=x) for x in range(1, forecastLength+1)]
    resultsByRegion = dataset.sirdSimulation(predictedParamsByRegion, simulationStartDate=forecastStartDate)

    targetRegions = [r for r in targetRegions if r in resultsByRegion.keys()]
    # Transform into dataframe
    resultsDf = dataset.transformIntoDf(resultsByRegion, forecastPeriod)

    if outputPlots is True:
        saveDir = outputDir

        targetRegions = ['LISBOA', 'PORTO', 'BEJA', 'GUARDA', 'COVILHA', 'AMADORA', 'ODIVELAS',
                         'CASTELO BRANCO', 'FARO', 'CASCAIS', 'AVEIRO', 'MONTIJO', 'CALDAS DA RAINHA',
                         'ESPINHO', 'MAIA', 'SINTRA', 'PENAFIEL', 'VISEU', 'SETUBAL', 'BRAGA', 'BARCELOS',
                         'LAGOS', 'ALCOBACA', 'NAZARE', 'LOURES', 'TORRES VEDRAS']

        # Plot parameter evoltuions
        logging.info('Plotting...')
        vz.smallMultiplesGenericWithProjected(paramsByRegion, predictedParamsByRegion, targetRegions,
                                              ['beta', 'gamma', 'delta'], saveDir, 'parameterEvolution')
        # Plot forecast
        historicPeriod = historicPeriod[historicPeriod.index(simulationOriginDay):]

        logging.info('Plotting...')
        vz.smallMultiplesForecastV2(df, resultsDf, historicPeriod, forecastPeriod, targetRegions, saveDir, 'forecast-new-cases')
        logging.info('Plotting...')
        vz.smallMultiplesForecastV2(df, resultsDf, historicPeriod, forecastPeriod, targetRegions, saveDir, 'forecast-new-cases-MA', maPeriod=movAvgInterval)

    return resultsDf






