import os
import sys
sys.path.append(os.getcwd() + '/..')
from argparse import ArgumentParser
import logging
import traceback
from datetime import datetime, timedelta
from os.path import join

import pandas as pd

from tasks import processNewData, generateDeathsAndRecoveries, makePredictions
from utils.loader import loadRefinedData, loadDeathsAndRecoveries
from basicPlots import basicPlot
import libs.pandasLib as pl
import libs.osLib as ol

if __name__ == '__main__':

        parser = ArgumentParser()

        parser.add_argument("-p", "--production", help="Whether to run entire pipeline or not (true when in production)",
                            type=bool, default=False)

        parser.add_argument("-rt", "--meanRecoveryTime",
                                help="Number of days that takes an infected person to recover", type=int, default=14)

        parser.add_argument("-fl", "--forecastLength", help="Number of days to predict in the future", type=int, default=15)

        parser.add_argument("-ss", "--simulationStartDate", help="Arbitrary date %Y-$m-%d (cannot be too close to the "
                                "beggining of the pandemic)", type=str, default='2020-05-01')

        parser.add_argument("-ma", "--movAvgInterval", help="Moving Average interval", type=int, default=7)

        parser.add_argument("-ppm", "--paramPredictMethod", help="", type=str, default='avg')


        args = parser.parse_args()

        simulationStartDate = pd.to_datetime(args.simulationStartDate, format='%Y-%m-%d')

        # Todo - before actually deploying i need to make sure this is all great - params are probably not in the right order

        # todo - set up logger fit for production
        logging.basicConfig(level=logging.INFO)

        logging.info(f'=== Running Full pipeline ===')

        logging.info(f'=== Stage 1 ===')
        previousLatestDate = pd.to_datetime(' 2020-12-13', format='%Y-%m-%d')    # todo - carefull here
        regionalDf, regions, timePeriod = processNewData(previousLatestDate=previousLatestDate)

        logging.info(f'=== Stage 2 ===')
        regionalDf, regions = generateDeathsAndRecoveries(regionalDf, regions, timePeriod, args.meanRecoveryTime)

        logging.info(f'=== Stage 3 ===')
        makePredictions(regionalDf, regions, timePeriod, simulationStartDate, timePeriod[-1], args.forecastLength, toDB=True)

        logging.info(f'=== All done ===')