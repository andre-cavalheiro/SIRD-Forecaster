import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback
from math import sqrt
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import wandb

from utils.customExceptions import NegativeActiveCases, ActiveCasesWithNaN, DeathsWithNaN, RecoveriesWithNaN


def calculateCasesPerRegion(newDf, latestOldDf, regions, days, locationAttr, dateAttr):

    # Calculate new values
    try:
        casesDf = newDf.value_counts([dateAttr, locationAttr])
        casesDf = casesDf.to_frame('new-cases')
        casesDf.index.names = ["Day", "Region"]
        casesDf = casesDf.sort_index()

        # Set day-region combinations without new cases to zero
        idx = pd.MultiIndex.from_product([days, regions], names=['Day', 'Region'])
        outputDf = pd.DataFrame(0, idx, ['new-cases'])
        outputDf.update(casesDf)

        logging.info('\t\t New cases done')

        for index, row in outputDf.iterrows():
            day, region = index[0], index[1]
            relevantDf = outputDf.loc[days[0]:day]
            relevantDf = relevantDf.loc[relevantDf.index.isin([region], level=1)]
            # totalCases = relevantDf['new-cases'].sum()
            totalCases = latestOldDf.loc[region, 'total-cases'] + relevantDf['new-cases'].sum()   # todo

            outputDf.at[(day, region), 'total-cases'] = totalCases

        logging.info('\t\t Total cases done')

        outputDf['new-cases'] = outputDf['new-cases'].astype(int)
        outputDf['total-cases'] = outputDf['total-cases'].astype(int)

    except Exception as ex:
        logging.error(traceback.format_exc())
        raise Exception('Breakdown')    # fixme

    return outputDf


def calculateDeathsAndRecoveries(regions, days, regionalDf, nationalDf, meanRecoveryTime=14, numDecimalPoints=3):
    idx = pd.IndexSlice

    # Init new columns
    for c in ['new-deaths', 'new-recovered', 'new-active-cases', 'total-active-cases']:
        regionalDf[c] = np.nan
        regionalDf[c] = regionalDf[c].astype(float)

    # todo - decide whether we should smooth here or not
    regionalDf = regionalDf.sort_index()

    failedRegions = []
    for region in regions:
        try:
            logging.info(f'Handling {region}')

            # Predict number of deaths and recoveries
            for it, day in enumerate(days):

                if it >= meanRecoveryTime:
                    # Calc death ratio based on national data
                    prior = days[it-meanRecoveryTime]
                    if nationalDf['novos_confirmados'][prior] == 0:
                        ratio = 0
                    else:
                        ratio = (nationalDf['novos_obitos'][day]/nationalDf['novos_confirmados'][prior]).round(numDecimalPoints)

                    priorCases = regionalDf.loc[(region, prior)]['new-cases']
                    if pd.isnull(priorCases):
                        priorCases = 0  # Day-Region combination doesnt exist in the dataset meaning no new cases.
                else:
                    # No deaths or recoveries in the first few days
                    ratio = 0
                    priorCases = 0

                assert(not pd.isnull(ratio))
                assert(not pd.isnull(priorCases))

                if ratio >= 1:
                    regionalDf.loc[(region, day), 'new-deaths'] = priorCases
                    regionalDf.loc[(region, day), 'new-recovered'] = 0
                else:
                    regionalDf.loc[(region, day), 'new-deaths'] = priorCases * ratio
                    regionalDf.loc[(region, day), 'new-recovered'] = priorCases - regionalDf.loc[(region, day)]['new-deaths']

            # Make sure everything went smoothly
            if regionalDf.loc[idx[region, :]]['new-deaths'].isna().any():
                raise DeathsWithNaN(region, regionalDf.loc[idx[region, :]]['new-deaths'])

            # Make sure everything wen smoothly
            if regionalDf.loc[idx[region, :]]['new-recovered'].isna().any():
                raise RecoveriesWithNaN(region, regionalDf.loc[idx[region, :]]['new-recovered'])

            # Calculate derivative measures accordingly
            regionalDf.loc[idx[region, :], 'new-active-cases'] = regionalDf.loc[idx[region, :], "new-cases"] - \
                                                                 regionalDf.loc[idx[region, :], "new-recovered"] - \
                                                                 regionalDf.loc[idx[region, :], "new-deaths"]

            regionalDf.loc[idx[region, :], 'total-recovered'] = regionalDf.loc[idx[region, :], 'new-recovered']\
                .cumsum().round(numDecimalPoints)

            regionalDf.loc[idx[region, :], 'total-deaths'] = regionalDf.loc[idx[region, :], 'new-deaths']\
                .cumsum().round(numDecimalPoints)

            regionalDf.loc[idx[region, :], 'total-active-cases'] = regionalDf.loc[idx[region, :], 'new-active-cases']\
                .cumsum().round(numDecimalPoints)
            R=1
            if regionalDf.loc[idx[region, :]]['total-active-cases'].isna().any():
                raise ActiveCasesWithNaN(region, regionalDf.loc[idx[region, :]]['total-active-cases'])  #MADALENA

            if any(regionalDf.loc[idx[region, :]]['total-active-cases'] < 0):
                raise NegativeActiveCases(region, regionalDf.loc[idx[region, :]]['total-active-cases'])

        except Exception as ex:
            logging.error(f'Failed {region}')
            logging.error(traceback.format_exc())
            failedRegions.append(region)

            # Sanity plots
            #################################################################
            '''
            import matplotlib.pyplot as plt
            import math
            columns = ['new-cases', 'new-recovered', 'new-deaths', 'new-active-cases', 'total-active-cases']

            numPlots, plotsPerRow = len(columns), 1
            numRows = math.ceil((numPlots / plotsPerRow))

            # multiple line plot
            f = plt.figure()
            num = 1

            for c in columns:
                ts = regionalDf.loc[idx[region, :], c].values
                pltoIndexes = [i for i in range(len(ts))]

                f.add_subplot(numRows, plotsPerRow, num)
                plt.plot(pltoIndexes, ts, label=c)
                num += 1
            plt.title(f'{region}')
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
            '''
            ##################################################################

    return regionalDf, failedRegions


def calculateNationalRecoveries(nationalDf, days, meanRecoveryTime=14, numDecimalPoints=3):
    nationalDf = nationalDf[['confirmados_novos', 'novos_obitos']]
    nationalDf.index = nationalDf.index.rename('Day')
    nationalDf = nationalDf.rename({
        'confirmados_novos': 'new-cases',
        'novos_obitos': 'new-deaths'
    }, axis=1)

    nationalDf = nationalDf.loc[nationalDf.index >= days[0]]
    nationalDf = nationalDf.loc[nationalDf.index <= days[-1]]

    # Fill in zeros for days that are not recorded
    for d in days:
        if d not in nationalDf.index:
            nationalDf = nationalDf.append(pd.Series(0, index=nationalDf.columns, name=d))

    openCases = 0
    for it, day in enumerate(days):
        if it >= meanRecoveryTime:
            prior = days[it - meanRecoveryTime]
            priorCases = nationalDf.loc[prior]['new-cases']
        else:
            # No deaths or recoveries in the first few days
            priorCases = 0
        curentDeaths = nationalDf.loc[day]['new-deaths']

        if priorCases >= curentDeaths:
            recoveries = priorCases - curentDeaths
            if openCases > 0:
                while recoveries > 0 and openCases > 0:
                    recoveries -= 1
                    openCases -= 1
                nationalDf.loc[day, 'new-recovered'] = recoveries
            nationalDf.loc[day, 'new-recovered'] = recoveries

        else:
            openCases += (curentDeaths-priorCases)
            recoveries = 0
            nationalDf.loc[day, 'new-recovered'] = recoveries

    # Make sure everything went smoothly
    if nationalDf['new-deaths'].isna().any():
        raise DeathsWithNaN('PORTUGAL', nationalDf['new-deaths'])

    # Make sure everything wen smoothly
    if nationalDf['new-recovered'].isna().any():
        raise RecoveriesWithNaN('PORTUGAL', nationalDf['new-recovered'])

    nationalDf['total-cases'] = nationalDf['new-cases'].cumsum().round(numDecimalPoints)
    nationalDf['total-recovered'] = nationalDf['new-recovered'].cumsum().round(numDecimalPoints)
    nationalDf['total-deaths'] = nationalDf['new-deaths'].cumsum().round(numDecimalPoints)
    nationalDf['new-active-cases'] = nationalDf["new-cases"] - nationalDf["new-recovered"] - nationalDf["new-deaths"]
    nationalDf['total-active-cases'] = nationalDf['new-active-cases'].cumsum().round(numDecimalPoints)

    if nationalDf['total-active-cases'].isna().any():
        raise ActiveCasesWithNaN('PORTUGAL', nationalDf['total-active-cases'])

    if any(nationalDf['total-active-cases'] < 0):
        raise NegativeActiveCases('PORTUGAL', nationalDf['total-active-cases'])

    # Sanity plots
    #################################################################
    """
    import matplotlib.pyplot as plt
    import math
    columns = ['new-cases', 'new-recovered', 'new-deaths', 'new-active-cases', 'total-active-cases']

    numPlots, plotsPerRow = len(columns), 1
    numRows = math.ceil((numPlots / plotsPerRow))

    # multiple line plot
    f = plt.figure()
    num = 1

    for c in columns:
        ts = nationalDf[c].values
        pltoIndexes = [i for i in range(len(ts))]

        f.add_subplot(numRows, plotsPerRow, num)
        plt.plot(pltoIndexes, ts, label=c)
        num += 1
    plt.title(f'PORTUGAL')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    """
    ##################################################################

    return nationalDf


def projectSIRDParamsForward(paramsByRegion, method, params, forecastLength, projectionParams):

    for region, payload in paramsByRegion.items():
        if method == 'last':
            for p in params:
                paramsByRegion[region][f'{p}-forecast'] = [payload[p].tolist()[-1] for _ in range(forecastLength)]
        elif method == 'avg':
            assert ('avgInterval' in projectionParams.keys())
            for p in params:
                paramsByRegion[region][f'{p}-forecast'] = [np.mean(payload[p].tolist()[-projectionParams['avgInterval']]) for _ in
                                                           range(forecastLength)]
        elif method == 'extrapolation':
            from scipy import interpolate
            for p in params:
                y = payload[p].tolist()
                x = [i for i in range(len(y))]
                f = interpolate.interp1d(x, y, fill_value="extrapolate")
                paramsByRegion[region][f'{p}-forecast'] = [f(i + len(y)).tolist() for i in range(forecastLength)]
        else:
            raise Exception(f'Unknown parameter projection method - {method}')

    return paramsByRegion


def pseudoCountsUpdate(paramsByRegion, projectedNationalParams, populationDf, k, params=['beta', 'gamma', 'delta']):

    for r, dt in paramsByRegion.items():

        #################################################################
        '''
        # Sanity plots
        import matplotlib.pyplot as plt
        import math
        f = plt.figure()
        numPlots, plotsPerRow = len(params), 1
        numRows = math.ceil((numPlots / plotsPerRow))
        num = 1
        '''
        #################################################################

        for p in params:
            old = paramsByRegion[r][p]

            numA = dt[p].multiply(populationDf[r])
            numB = projectedNationalParams['PORTUGAL'][p].multiply(k)
            denom = k + populationDf[r]
            paramsByRegion[r][f'{p}'] = (numA + numB) / (denom)

            #################################################################
            '''
            # Sanity plots
            f.add_subplot(numRows, plotsPerRow, num)
            num += 1

            idx = [i for i in range(len(old))]

            plt.plot(idx, projectedNationalParams['PORTUGAL'][p], label='national')
            plt.plot(idx, old, label='old')
            plt.plot(idx, paramsByRegion[r][p], label='new')

        plt.title(f'{r} ({populationDf[r]}), k={k}')
        plt.legend()
        plt.show()
        plt.close()'''
        ##################################################################
    return paramsByRegion


def evaluatePredictionsOLD(regionalDf, forecastDf, forecastPeriod, regions2Plot):
    results = {}

    for day in forecastPeriod:
        if not regionalDf.index.isin([forecastPeriod[-1]], level=1).any():
            break

        targets, predictions = [], []

        for r in regions2Plot:
            x = regionalDf.loc[(r, day), 'new-cases']
            targets.append(x)
            y = forecastDf.loc[(day, r), 'predictedCases']
            predictions.append(y)

        results[day] = {}
        results[day]['targets'] = targets
        results[day]['predictions'] = predictions
        results[day]['R2'] = r2_score(targets, predictions)
        results[day]['RMSE'] = sqrt(mean_squared_error(targets, predictions))

    results['fullPeriod'] = {}
    results['fullPeriod']['origDate'] = forecastPeriod[0]
    results['fullPeriod']['targets'] = [x for d in forecastPeriod for x in results[d]['targets']]
    results['fullPeriod']['predictions'] = [x for d in forecastPeriod for x in results[d]['predictions']]
    results['fullPeriod']['R2'] = r2_score(results['fullPeriod']['targets'], results['fullPeriod']['predictions'])
    results['fullPeriod']['RMSE'] = sqrt(mean_squared_error(results['fullPeriod']['targets'], results['fullPeriod']['predictions']))

    '''
    R2PerReg = []
    idx = pd.IndexSlice
    for r in regions2Plot:
        targets = regionalDf.loc[r].loc[forecastPeriod[0]:forecastPeriod[-1]]['new-cases']
        predictions = forecastDf.xs(r, level=1).loc[forecastPeriod[0]:forecastPeriod[-1]]['predictedCases']
        r2 = r2_score(targets, predictions)
        R2PerReg.append(r2)
    results['fullPeriod']['R2'] = sum(R2PerReg)/len(R2PerReg)
    '''

    return results


def evaluatePredictionsByRegionClassOLD(regionalDf, populationDf, forecastDf, forecastPeriod, regions2Plot, labels=["verySmall", "small", "big", "veryBig"]):

    with open('../data/regionMapping.json') as f:
        regionMapping = json.load(f)
    populationDf = populationDf[populationDf.index.isin(regionMapping.keys())]

    regionTypes = pd.qcut(populationDf, 4, labels=labels)
    results = {l: {} for l in labels}

    for day in forecastPeriod:
        if not regionalDf.index.isin([forecastPeriod[-1]], level=1).any():
            break

        targets, predictions = {l: [] for l in labels}, {l: [] for l in labels}

        for r in regions2Plot:
            x = regionalDf.loc[(r, day), 'new-cases']
            targets[regionTypes[r]].append(x)
            y = forecastDf.loc[(day, r), 'predictedCases']
            predictions[regionTypes[r]].append(y)

        for l in labels:
            results[l][day] = {}
            results[l][day]['targets'] = targets[l]
            results[l][day]['predictions'] = predictions[l]
            results[l][day]['R2'] = r2_score(targets[l], predictions[l])
            results[l][day]['RMSE'] = sqrt(mean_squared_error(targets[l], predictions[l]))

    for l in labels:
        results[l]['fullPeriod'] = {}
        results[l]['fullPeriod']['origDate'] = forecastPeriod[0]
        results[l]['fullPeriod']['targets'] = [x for d in forecastPeriod for x in results[l][d]['targets']]
        results[l]['fullPeriod']['predictions'] = [x for d in forecastPeriod for x in results[l][d]['predictions']]
        results[l]['fullPeriod']['R2'] = r2_score(results[l]['fullPeriod']['targets'], results[l]['fullPeriod']['predictions'])
        results[l]['fullPeriod']['RMSE'] = sqrt(mean_squared_error(results[l]['fullPeriod']['targets'], results[l]['fullPeriod']['predictions']))

    return results


def evaluatePredictions(regionalDf, forecastDf, populationDf, historicPeriod, forecastPeriod, regions2Plot, maPeriod, labels=["verySmall", "small", "big", "veryBig"]):

    id = forecastPeriod[0]
    history = {}
    # Apply moving average for the entire period
    for r in regions2Plot:
        history[r] = regionalDf.loc[r].loc[historicPeriod[0]:historicPeriod[-1]]['new-cases'].rolling(maPeriod, min_periods=1).mean()

    # Load region types (divided by population size)
    with open('../data/regionMapping.json') as f:
        regionMapping = json.load(f)
    populationDf = populationDf[populationDf.index.isin(regionMapping.keys())]
    regionTypes = pd.qcut(populationDf, len(labels), labels=labels)

    # Extract Targets and predictions
    R2s, RMSEs, resultsPerRegionType, allResults = [], [], {l: {'targets': [], 'predictions': []} for l in labels}, {'targets': [], 'predictions': []}
    for r in regions2Plot:
        target = history[r].loc[forecastPeriod[0]:forecastPeriod[-1]]
        prediction = forecastDf.xs(r, level=1).loc[forecastPeriod[0]:forecastPeriod[-1]]['predictedCases']
        resultsPerRegionType[regionTypes[r]]['targets'].append(target)
        resultsPerRegionType[regionTypes[r]]['predictions'].append(prediction)
        allResults['targets'].append(target)
        allResults['predictions'].append(prediction)


    for l, dt in resultsPerRegionType.items():
        r2 = r2_score(dt['targets'], dt['predictions'], multioutput='variance_weighted')
        rmse = mean_squared_error(dt['targets'], dt['predictions'], squared=False)
        wandb.log({
                f'{l}/R2/{id}': r2,
                f'{l}/RMSE/{id}': rmse,
            })

    globalR2 = r2_score(allResults['targets'], allResults['predictions'], multioutput='variance_weighted')
    globalRMSE = mean_squared_error(allResults['targets'], allResults['predictions'], squared=False)

    wandb.log({
        f'R2/{id}': globalR2,
        f'RMSE/{id}': globalRMSE,
    })
