import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback

import pandas as pd
import json

from utils.loader import loadNewDf, loadOldLatestDf, loadRefinedData, loadPopulation, pullNationalData
from utils.saver import updateCasesPerRegion, updateRecoveredAndDead, updateForecast
from utils.calculations import calculateCasesPerRegion, calculateDeathsAndRecoveries
from utils.forecast import sirdForecast
from utils.customExceptions import NegativeActiveCases

def processNewData(locationAttr='municipality_of_ocurrence', dateAttr='confirmation_date1',
                   inputAttr='input_date', previousLatestDate=None):

    # todo - make sure this is right
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


def makePredictions(regionalDf, regions, historicPeriod, simulationStartDate, forecastStartDate, forecastLength,
                    toDB=True, outputPlots=False, movAvgInterval=7, outputDir='.'):

    populationDf = loadPopulation()

    forecastDf = sirdForecast(regionalDf, populationDf, regions, historicPeriod, simulationStartDate, forecastStartDate,
                              forecastLength, 'avg', outputPlots=outputPlots, movAvgInterval=movAvgInterval,
                              outputDir=outputDir)

    logging.info(f'\t Performed prediction successfully \t')

    with open('../data/regionMapping.json') as f:
        regionMapping = json.load(f)

    forecastDf = forecastDf.reset_index()
    forecastDf['originalRegion'] = ''
    forecastDf['originalRegion'] = forecastDf.apply(
        lambda x: regionMapping[x['Region']] if x['Region'] in regionMapping.keys() else x['Region'], axis=1)

    if toDB is True:
        updateForecast(forecastDf)
        logging.info(f'\t Updated predictions to DB \t')

    return forecastDf

