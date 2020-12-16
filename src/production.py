import os
import sys
sys.path.append(os.getcwd() + '/..')
from argparse import ArgumentParser
import logging
import traceback
from datetime import datetime, timedelta
from os.path import join

import pandas as pd

from utils.tasks import processNewData, generateDeathsAndRecoveries, makePredictions
from utils.loader import loadRefinedData, loadDeathsAndRecoveries
from basicPlots import basicPlot
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

        parser.add_argument("-fl", "--forecastLength",
                                help="Number of days to predict in the future", type=int, default=15)

        parser.add_argument("-ss", "--simulationStartDate",
                                help="Arbitrary date %Y-$m%d (cannot be too close to the beggining of the pandemic) - also, make sure that it's after the day from which we have recoveries/deaths",
                            type=str, default='2020-05-01')

        parser.add_argument("-ma", "--movAvgInterval",
                                help="Moving Average interval", type=int, default=7)

        args = parser.parse_args()

        simulationStartDate = pd.to_datetime(args.simulationStartDate, format='%Y-%m-%d')

        # todo - set up logger fit for production
        logging.basicConfig(level=logging.INFO)

        logging.info(f'=== Running selected stages ===')

        if args.stage1 is True:
            logging.info(f'=== Stage 1 ===')
            regionalDf, regions, timePeriod = processNewData()
        else:
            regionalDf, regions, timePeriod = loadRefinedData()     # todo - time period here is not trivial
            logging.info(f'=== Loaded refined data from DB ===')

        if args.stage2 is True:
            logging.info(f'=== Stage 2 ===')
            regionalDf, regions = generateDeathsAndRecoveries(regionalDf, regions, timePeriod, args.meanRecoveryTime)
        else:
            generatedDf, regions = loadDeathsAndRecoveries(regions)
            regionalDf = pd.concat([regionalDf, generatedDf], axis=1)
            logging.info(f'=== Loaded deaths and recoveries from DB===')

        if args.stage3 is True:
            logging.info(f'=== Stage 3 ===')

            forecastStartDate = timePeriod[-1]

            outputDir = ol.getDir('../data/production', (forecastStartDate+timedelta(days=1)).strftime("%Y-%m-%d"))

            predictionDf = makePredictions(regionalDf, regions, timePeriod, simulationStartDate,
                                           forecastStartDate, args.forecastLength, toDB=False, outputPlots=True,
                                           movAvgInterval=args.movAvgInterval, outputDir=outputDir)

            pl.save(predictionDf, 'csv', join(outputDir, f'predictions'))
        else:
            pass
