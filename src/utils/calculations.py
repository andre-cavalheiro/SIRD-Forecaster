import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback
from math import sqrt
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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
            if day == days[0]:
                totalCases = outputDf.loc[day, region]['new-cases']
            else:
                relevantDf = outputDf.loc[days[0]:day]
                relevantDf = relevantDf.loc[relevantDf.index.isin([region], level=1)]
                totalCases = relevantDf['new-cases'].sum()
                # totalCases = latestOldDf.loc[region, 'total-cases'] + relevantDf['new-cases'].sum()   # todo

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
        regionalDf[c] = np.nan       # fixme - this should be NaN when I'm completly confident with this function
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

                if pd.isnull(regionalDf.loc[(region, day), 'new-deaths']) or pd.isnull(regionalDf.loc[(region, day), 'new-recovered']):
                    r=1
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

            if regionalDf.loc[idx[region, :]]['total-active-cases'].isna().any():
                raise ActiveCasesWithNaN(region, regionalDf.loc[idx[region, :]]['total-active-cases'])

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


def evaluatePredictions(regionalDf, forecastDf, forecastPeriod, regions2Plot):
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
        #results[day]['R2'] = r2_score(targets, predictions)
        #results[day]['RMSE'] = sqrt(mean_squared_error(targets, predictions))

    results['fullPeriod'] = {}
    results['fullPeriod']['origDate'] = forecastPeriod[-1]
    results['fullPeriod']['targets'] = [x for d in forecastPeriod for x in results[d]['targets']]
    results['fullPeriod']['predictions'] = [x for d in forecastPeriod for x in results[d]['predictions']]
    #results['fullPeriod']['R2'] = r2_score(results['fullPeriod']['targets'], results['fullPeriod']['predictions'])
    #results['fullPeriod']['RMSE'] = sqrt(mean_squared_error(results['fullPeriod']['targets'], results['fullPeriod']['predictions']))

    return results


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