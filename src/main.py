import os
import sys
sys.path.append(os.getcwd() + '/..')
from argparse import ArgumentParser
import logging
import traceback
from datetime import datetime, timedelta
from os.path import join
import numpy as np

import wandb
import pandas as pd
import json

from tasks import processNewData, generateDeathsAndRecoveries, makePredictions
from utils.loader import loadRefinedData, loadDeathsAndRecoveries
import libs.pandasLib as pl
import libs.osLib as ol

if __name__ == '__main__':

        parser = ArgumentParser()

        parser.add_argument("-s1", "--stage1",
                                help="Whether to run stage 1 from ground up (if false, load from DB)", type=bool,
                                default=False)

        parser.add_argument("-s2", "--stage2",
                                help="Whether to run stage 2 from ground up (if false, load from DB)", type=bool,
                                default=False)

        parser.add_argument("-s3", "--stage3",
                                help="Whether to run stage 3 from ground up (if false, load from DB)", type=bool,
                                default=False)

        parser.add_argument("-rt", "--meanRecoveryTime",
                                help="Number of days that takes an infected person to recover", type=int, default=14)

        parser.add_argument("-fl", "--forecastLength", help="Number of days to predict in the future", type=int, default=7)

        parser.add_argument("-ss", "--simulationStartDate", help="Arbitrary date %Y-$m-%d (cannot be too close to the "
                                "beggining of the pandemic)", type=str, default='2020-05-01')

        parser.add_argument("-ma", "--movAvgInterval", help="Moving Average interval", type=int, default=14)

        parser.add_argument("-ppm", "--paramPredictMethod", help="", type=str, default='extrapolation')
        parser.add_argument("-ppi", "--paramPredictInterval", help="", type=int, default=3)

        parser.add_argument("-pc", "--pseudoCounts", type=bool, default=True)
        parser.add_argument("-k", "--pseudoCountsK", type=float, default=10000)

        parser.add_argument("-c", "--constantParams", type=bool, default=False)
        parser.add_argument("-bc", "--betaConst", type=float, default=0.08)
        parser.add_argument("-gc", "--gammaConst", type=float, default=0.08)
        parser.add_argument("-dc", "--deltaConst", type=float, default=0.003)

        args = parser.parse_args()

        simulationStartDate = pd.to_datetime(args.simulationStartDate, format='%Y-%m-%d')

        with open('../data/regionMapping.json') as f:
            regionMapping = json.load(f)

        regions = list(regionMapping.keys())
        logging.basicConfig(level=logging.INFO)

        logging.info(f'=== Running selected stages ===')

        if args.stage1 is True:
            logging.info(f'=== Stage 1 ===')
            regionalDf, timePeriod = processNewData(regions)
        else:
            regionalDf, timePeriod = loadRefinedData(fromOS=False)
            logging.info(f'=== Loaded refined data from DB ===')

        if args.stage2 is True:
            logging.info(f'=== Stage 2 ===')
            regionalDf, regions = generateDeathsAndRecoveries(regionalDf, regions, timePeriod, args.meanRecoveryTime)
        else:
            generatedDf, regions = loadDeathsAndRecoveries(regions, fromOS=False)
            regionalDf = pd.concat([regionalDf, generatedDf], axis=1)
            logging.info(f'=== Loaded deaths and recoveries from DB===')

        if args.stage3 is True:
            logging.info(f'=== Stage 3 ===')
            timePeriod = timePeriod[timePeriod.index(simulationStartDate):]
            simDates = [timePeriod[-1]]
            print(simDates)
            # For paper results
            minDay, maxDay = pd.to_datetime('2021-02-20', format='%Y-%m-%d'), timePeriod[-1]
            print(minDay, maxDay)
            # simDates = [minDay + timedelta(days=x) for x in range((maxDay - minDay).days + 1)]

            auxDay = minDay
            simDates = [minDay]
            while auxDay + timedelta(days=7) < maxDay:
                simDates.append(auxDay+ timedelta(days=7))
                auxDay = auxDay+ timedelta(days=7)

            rootDir = '../data'

            projectionParams = {'method': args.paramPredictMethod, 'avgInterval': args.paramPredictInterval,
                                'pseudoCounts': args.pseudoCounts, 'pseudoCountsK': args.pseudoCountsK}
            sirdParams = {'movAvgInterval': args.movAvgInterval, 'applyConstParams': args.constantParams,
                          'constParams': {'beta': args.betaConst, 'gamma': args.gammaConst, 'delta': args.deltaConst}}

            outputDf = None
            print('===== Starting forcast =====')
            for forecastStartDate in simDates:
                logging.info(f'*** {forecastStartDate} ***')
                outputDir = ol.getDir(rootDir, (forecastStartDate+timedelta(days=7)).strftime("%Y-%m-%d"))
                print(outputDir)
                # outputDir = None
                predictionDf = makePredictions(regionalDf, regions, timePeriod, forecastStartDate, args.forecastLength,
                                               projectionParams, sirdParams, toDB=False, outputPlots=False,
                                               outputDir=outputDir)

                # print('OUT OF THE LOOOP - WRITING FILE')

                # pl.save(predictionDf, 'csv', join(outputDir, f'predictions'))
                # pl.save(R0DF, 'csv', join(outputDir, f'R0DF'))

                # For paper results - Keep only last day per prediction
                startDate = predictionDf.index.get_level_values(0).min()
                latestDate = predictionDf.index.get_level_values(0).max()
                print(f'Appending from {startDate} to {latestDate}')
                '''
                auxDf = predictionDf.loc[latestDate]
                auxDf['Day'] = latestDate'''
                print(f'CURRENT LATEST DATE: {latestDate}')
                if outputDf is None:
                    outputDf = predictionDf
                else:
                    outputDf = pd.concat([outputDf, predictionDf], axis=0)
                    r=1

                print(outputDf.tail())

                pl.save(outputDf, 'csv', join(rootDir, f'predictions-long-dataset-march'))
        else:
            pass
