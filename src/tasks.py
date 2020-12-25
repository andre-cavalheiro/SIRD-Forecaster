import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback

import pandas as pd
import json
from datetime import datetime, timedelta

from utils.loader import loadNewDf, loadOldLatestDf, loadRefinedData, loadPopulation, pullNationalData
from utils.saver import updateCasesPerRegion, updateRecoveredAndDead, updateForecast
from utils.calculations import calculateCasesPerRegion, calculateDeathsAndRecoveries, evaluatePredictions, calculateNationalRecoveries
from utils.forecast import sirdForecast
from utils.customExceptions import NegativeActiveCases, ActiveCasesWithNaN
import libs.visualization as vz
from libs.utils import addOriginalRegionLabels
from utils.sirdWrapper import nationalWrapper
from utils.forecast import projectParamsForwards

def processNewData(locationAttr='municipality_of_ocurrence', dateAttr='confirmation_date1',
                   inputAttr='input_date', previousLatestDate=None):

    latestOldDf = None
    # latestOldDf, previousLatestDate = loadOldLatestDf()
    logging.info(f'\t Previous data went only until: {previousLatestDate} \t')

    newDf, regions, newPeriod = loadNewDf(previousLatestDate, locationAttr, dateAttr, inputAttr)
    logging.info(f'\t Received regional data - most recent date: {newPeriod[-1]} \t')

    newDf = calculateCasesPerRegion(newDf, latestOldDf, regions, newPeriod, locationAttr, dateAttr)
    logging.info(f'\t Calculated infections per region per day \t')

    updateCasesPerRegion(newDf)
    logging.info(f'\t Updated DB with new infections \t')

    return newDf, regions, newPeriod


def generateDeathsAndRecoveries(regionalDf, regions, timePeriod, meanRecoveryTime):
    nationalDf, mostRecentDate = pullNationalData(regions, timePeriod)  # todo - verify dates are sufficient
    logging.info(f'\t Pulled national data - most recent date: {mostRecentDate} \t')

    regionalDf, failedRegions = calculateDeathsAndRecoveries(regions, timePeriod, regionalDf=regionalDf,
                                                             nationalDf=nationalDf,
                                                             meanRecoveryTime=meanRecoveryTime)
    regions = [r for r in regions if r not in failedRegions]
    logging.info(f'\t Generated deaths and recoveries \t')

    updateRecoveredAndDead(regionalDf)
    logging.info(f'\t Sent them do DB \t')

    return regionalDf, regions


def makePredictions(regionalDf, regions, historicPeriod, forecastStartDate, forecastLength,
                    paramPredictMethod, movAvgInterval, applyConstantParams, pseudoCounts,
                    toDB=False, outputPlots=False, outputDir='.', constantParams={}, paramPredictInterval=None):

    populationDf = loadPopulation()

    # Remove regions whose population amount we don't have
    regionsToDel = [r for r in regions if r not in populationDf.index]
    regions = [r for r in regions if r not in regionsToDel]
    if len(regionsToDel) > 0:
        logging.warning(f'Population dataset doesn\'t include {len(regionsToDel)} regions (ignoring them):')
        logging.warning(f'{regionsToDel}')

    # Define temporal periods
    sirdPeriod = historicPeriod[:historicPeriod.index(forecastStartDate)+1]
    forecastPeriod = [forecastStartDate + timedelta(days=x) for x in range(1, forecastLength+1)]

    if pseudoCounts is True:
        nationalDf, _ = pullNationalData(regions, historicPeriod)  # todo - verify dates are sufficient
        nationalDf = calculateNationalRecoveries(nationalDf, historicPeriod)

        nationalDf['Region'] = 'PORTUGAL'
        nationalDf = nationalDf.reset_index()
        # nationalDf = (nationalDf.set_index(['Region', 'data']).sort_index())
        nw = nationalWrapper(nationalDf, sirdPeriod[0], sirdPeriod)
        # Apply moving average to time series
        nw.movingAvg(["new-cases", "new-recovered", "new-deaths", "total-cases", "total-recovered", "total-deaths", "total-active-cases"], movAvgInterval)
        # Use historic data to calculate parameters
        nw.calculateSIRDParametersByDay()
        # Smooth parameter time series
        nw.smoothSIRDParameters()
        nationalParams = nw.getParams()

    # Perform predictions
    forecastDf, paramsByReg, maxParam = sirdForecast(regionalDf, populationDf, regions, sirdPeriod, forecastPeriod,
                                              paramPredictMethod, movAvgInterval, applyConstantParams, pseudoCounts,
                                             constantParams=constantParams, paramPredictInterval=paramPredictInterval,
                                                     nationalParams=nationalParams)

    logging.info(f'\t Performed prediction successfully \t')

    forecastDf = addOriginalRegionLabels(forecastDf)

    if toDB is True:
        updateForecast(forecastDf)
        logging.info(f'\t Updated predictions to DB \t')

    if outputPlots is True:
        saveDir = outputDir
        """"
        regions2Plot = ['LISBOA', 'PORTO', 'BEJA', 'GUARDA', 'COVILHA', 'AMADORA', 'ODIVELAS',
                         'CASTELO BRANCO', 'FARO', 'CASCAIS', 'AVEIRO', 'MONTIJO', 'CALDAS DA RAINHA',
                         'ESPINHO', 'MAIA', 'SINTRA', 'PENAFIEL', 'VISEU', 'SETUBAL', 'BRAGA', 'BARCELOS',
                         'LAGOS', 'ALCOBACA', 'NAZARE', 'LOURES', 'TORRES VEDRAS', 'GUIMARAES', 'CRATO', 'BARCELOS',
                        'LOUSADA', 'RIO MAIOR', 'AGUEDA', 'FIGUEIRA DA FOZ', 'TONDELA', 'OBIDOS', 'OURIQUE', 'ESPINHO',
                        'FAFE', 'NISA', 'MARVAO']
        regions2Plot = ['LISBOA', 'OEIRAS', 'ALCOCHETE', 'MOITA', 'VIMIOSO', 'BRAGANCA', 'MERTOLA', 'SERPA', 'ALENQUER',
                        'RIO MAIOR', 'EVORA', 'IDANHA-A-NOVA', 'PENAMACOR', 'PINHEL', 'FIGUEIRA DE CASTELO RODRIGO']
        """

        regions2Plot = regions

        logging.info('Updating to wandb...')
        results = evaluatePredictions(regionalDf, forecastDf, forecastPeriod, regions2Plot)
        # vz.logToWandb(results)

        # Plot correlation
        logging.info('Plotting...')
        #vz.forecastCorrelation(results, forecastPeriod, saveDir, 'scatter')

        # Plot parameter evoltuions
        #logging.info('Plotting...')
        #vz.smallMultiplesGenericWithProjected(paramsByReg, regions2Plot, ['beta', 'gamma', 'delta'], forecastPeriod,
        #                                      maxParam, saveDir, 'parameterEvolution')

        # Plot forecast
        #logging.info('Plotting...')
        #vz.smallMultiplesForecastV2(regionalDf, forecastDf, historicPeriod, forecastPeriod, regions2Plot, saveDir, 'forecast-new-cases')
        logging.info('Plotting...')
        vz.smallMultiplesForecastV2(regionalDf, forecastDf, historicPeriod, forecastPeriod, regions2Plot, saveDir, 'forecast-new-cases-MA', maPeriod=movAvgInterval)
    return forecastDf

