import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback

import numpy as np
import pandas as pd
import json
from datetime import date, datetime, timedelta

from utils.loader import loadNewDf, loadOldLatestDf, loadRefinedData, loadPopulation, pullNationalData
from utils.saver import updateCasesPerRegion, updateRecoveredAndDead, updateForecast
from utils.calculations import calculateCasesPerRegion, calculateDeathsAndRecoveries, evaluatePredictions, calculateNationalRecoveries
from utils.customExceptions import NegativeActiveCases, ActiveCasesWithNaN
import libs.visualization as vz
from libs.utils import addOriginalRegionLabels
from sird.sirdNationalWrapper import nationalWrapper
from sird.forecast import nationalSIRD, initSimulation, setForecastingParams, forecast


def processNewData(regions, locationAttr='municipality_of_ocurrence', dateAttr='confirmation_date1',
                   inputAttr='input_date', previousLatestDate=None, ):

    # latestOldDf, previousLatestDate = pd.to_datetime('2020-03-02', format='%Y-%m-%d')
    latestOldDf, previousLatestDate = loadOldLatestDf()

    logging.info(f'\t Previous data went only until: {previousLatestDate} \t')

    newDf, newPeriod = loadNewDf(previousLatestDate, locationAttr, dateAttr, inputAttr)
    logging.info(f'\t Received regional data - most recent date: {newPeriod[-1]} \t')

    newDf = calculateCasesPerRegion(newDf, latestOldDf, regions, newPeriod, locationAttr, dateAttr)
    logging.info(f'\t Calculated infections per region per day \t')

    updateCasesPerRegion(newDf)
    logging.info(f'\t Updated DB with new infections \t')

    return newDf, newPeriod


def generateDeathsAndRecoveries(regionalDf, regions, timePeriod, meanRecoveryTime):
    nationalDf, mostRecentDate = pullNationalData(regions, timePeriod)  # todo - verify dates are sufficient
    logging.info(f'\t Pulled national data - most recent date: {mostRecentDate} \t')

    regionalDf, failedRegions = calculateDeathsAndRecoveries(regions, timePeriod, regionalDf=regionalDf, nationalDf=nationalDf, meanRecoveryTime=meanRecoveryTime)
    regions = [r for r in regions if r not in failedRegions]
    logging.info(f'\t Generated deaths and recoveries \t')

    updateRecoveredAndDead(regionalDf)
    logging.info(f'\t Sent them do DB \t')

    return regionalDf, regions


def makePredictions(regionalDf, regions, historicPeriod, forecastStartDate, forecastLength,
                    projectionParams, sirdParams, toDB=False, outputPlots=False, outputDir='.', ):

    populationDf = loadPopulation()

    # Remove regions whose population amount we don't have
    regionsToDel = [r for r in regions if r not in populationDf.index]
    regions = [r for r in regions if r not in regionsToDel]
    if len(regionsToDel) > 0:
        logging.warning(f'Population dataset doesn\'t include {len(regionsToDel)} regions (ignoring them):')
        logging.warning(f'{regionsToDel}')

    # Define temporal periods
    sirdPeriod = historicPeriod[:historicPeriod.index(forecastStartDate)+2]
    forecastPeriod = [forecastStartDate + timedelta(days=x) for x in range(1, forecastLength+1)]

    print('FITTING TO HISTORY')

    # Fit sird parameters to historical data
    wp = initSimulation(regionalDf, populationDf, regions, sirdPeriod, sirdParams['applyConstParams'], sirdParams['constParams'], sirdParams['movAvgInterval'])

    # Project SIRD parameters forward
    # If necessary run national model
    nationalParams = nationalSIRD(regions, historicPeriod, sirdPeriod, sirdParams['movAvgInterval'])

    paramsByRegion, maxParam = setForecastingParams(wp, projectionParams['method'], forecastPeriod, projectionParams=projectionParams, pseudoCounts=projectionParams['pseudoCounts'], nationalParams=nationalParams, populationDf=populationDf)

    #with open('../data/regionMapping.json') as f:
    #    regionMapping = json.load(f)

    # r0ByRegion = {regionMapping[k]: v['R0'] for k, v in paramsByRegion.items()}
    # r0Df = pd.DataFrame.from_dict(r0ByRegion,orient='index').transpose()

    # Run future simulation
    predictedParams = {r: {c: v[f'{c}-forecast'] for c in ['beta', 'gamma', 'delta']} for r, v in paramsByRegion.items()}  # Change names to match plot requirements
    forecastDf = forecast(wp, predictedParams, forecastPeriod)
    logging.info(f'\t Performed prediction successfully \t')
    # Add regional labels to match CERENA requirements
    forecastDf = addOriginalRegionLabels(forecastDf)

    # Save to DB
    if toDB is True:
        updateForecast(forecastDf)
        logging.info(f'\t Updated predictions to DB \t')

    # Make plots
    if outputPlots is True:
        regions2Plot = regions

        logging.info('Updating to wandb...')
        # evaluatePredictions(regionalDf, forecastDf, populationDf, historicPeriod, forecastPeriod, regions2Plot, sirdParams['movAvgInterval'])


        # results = evaluatePredictions(regionalDf, forecastDf, forecastPeriod, regions2Plot)
        # resultsByRegionType = evaluatePredictionsByRegionClass(regionalDf, populationDf, forecastDf, forecastPeriod, regions2Plot)
        # vz.logToWandb(results)
        # vz.logToWandbByRegionType(resultsByRegionType)

        # Plot correlation
        # logging.info('Plotting...')
        # vz.forecastCorrelation(results, forecastPeriod, outputDir, 'scatter')

        # Load region types (divided by population size)
        labels = ["verySmall", "small", "big", "veryBig"]
        with open('../data/regionMapping.json') as f:
            regionMapping = json.load(f)
        populationDf = populationDf[populationDf.index.isin(regionMapping.keys())].sort_values(ascending=False)
        regionTypes = pd.qcut(populationDf, len(labels), labels=labels)

        for l in ["verySmall", "small", "big", "veryBig"]:
            regions2Plot = regionTypes[regionTypes==l][:10].index.tolist() + regionTypes[regionTypes==l][-10:].index.tolist()
            # Plot parameter evoltuions
            logging.info('Plotting...')
            vz.smallMultiplesGenericWithProjected(paramsByRegion, regions2Plot, ['beta', 'gamma', 'delta'], forecastPeriod, maxParam, outputDir, f'parameterEvolution-{l}')

            logging.info('Plotting...')
            vz.smallMultiplesForecastV2(regionalDf, forecastDf, historicPeriod, forecastPeriod, regions2Plot, outputDir, f'forecast-MA-{l}', maPeriod=sirdParams['movAvgInterval'])

    return forecastDf
