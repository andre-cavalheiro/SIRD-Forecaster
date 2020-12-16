import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback

import pandas as pd
import numpy as np

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
        regionalDf[c] = 0       # fixme - this should be NaN when I'm completly confident with this function
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
            # Make sure everything wen smoothly
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
