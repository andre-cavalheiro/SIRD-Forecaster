import os
import sys
sys.path.append(os.getcwd() + '/..')
from argparse import ArgumentParser
import logging
import traceback
from datetime import datetime, timedelta
from os.path import join

import wandb
import pandas as pd

from tasks import processNewData, generateDeathsAndRecoveries, makePredictions
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

        parser.add_argument("-fl", "--forecastLength", help="Number of days to predict in the future", type=int, default=7)

        parser.add_argument("-ss", "--simulationStartDate", help="Arbitrary date %Y-$m-%d (cannot be too close to the "
                                "beggining of the pandemic)", type=str, default='2020-05-01')

        parser.add_argument("-ma", "--movAvgInterval", help="Moving Average interval", type=int, default=14)

        parser.add_argument("-ppm", "--paramPredictMethod", help="", type=str, default='avg')
        parser.add_argument("-ppi", "--paramPredictInterval", help="", type=int, default=7)

        parser.add_argument("-pc", "--pseudoCounts", type=bool, default=True)
        parser.add_argument("-c", "--constantParams", type=bool, default=False)
        parser.add_argument("-bc", "--betaConst", type=float, default=0.08)
        parser.add_argument("-gc", "--gammaConst", type=float, default=0.08)
        parser.add_argument("-dc", "--deltaConst", type=float, default=0.003)

        args = parser.parse_args()

        simulationStartDate = pd.to_datetime(args.simulationStartDate, format='%Y-%m-%d')

        logging.basicConfig(level=logging.INFO)

        logging.info(f'=== Running selected stages ===')

        if args.stage1 is True:
            logging.info(f'=== Stage 1 ===')
            regionalDf, regions, timePeriod = processNewData()
        else:
            regionalDf, regions, timePeriod = loadRefinedData(fromOS=False)
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
            #  = timePeriod[timePeriod.index(simulationStartDate):]

            simDates = [
                # pd.to_datetime('2020-08-31', format='%Y-%m-%d'),
                # pd.to_datetime('2020-09-16', format='%Y-%m-%d'),
                # pd.to_datetime('2020-10-01', format='%Y-%m-%d'),
                pd.to_datetime('2020-10-16', format='%Y-%m-%d'),
                pd.to_datetime('2020-10-31', format='%Y-%m-%d'),
                pd.to_datetime('2020-11-15', format='%Y-%m-%d'),
                pd.to_datetime('2020-11-30', format='%Y-%m-%d'),
                # pd.to_datetime('2020-12-06', format='%Y-%m-%d')
            ]

            """
            regions = ['LISBOA', 'PORTO', 'BEJA', 'GUARDA', 'COVILHA', 'AMADORA', 'ODIVELAS',
                         'CASTELO BRANCO', 'FARO', 'CASCAIS', 'AVEIRO', 'MONTIJO', 'CALDAS DA RAINHA',
                         'ESPINHO', 'MAIA', 'SINTRA', 'PENAFIEL', 'VISEU', 'SETUBAL', 'BRAGA', 'BARCELOS',
                         'LAGOS', 'ALCOBACA', 'NAZARE', 'LOURES', 'TORRES VEDRAS', 'GUIMARAES', 'CRATO', 'BARCELOS',
                        'LOUSADA', 'RIO MAIOR', 'AGUEDA', 'FIGUEIRA DA FOZ', 'TONDELA', 'OBIDOS', 'OURIQUE', 'ESPINHO',
                        'FAFE', 'NISA', 'MARVAO', 'OEIRAS', 'ALCOCHETE', 'MOITA', 'VIMIOSO', 'BRAGANCA', 'MERTOLA',
                        'SERPA', 'ALENQUER', 'RIO MAIOR', 'EVORA', 'IDANHA-A-NOVA', 'PENAMACOR', 'PINHEL',
                        'FIGUEIRA DE CASTELO RODRIGO']
            """
            regions =['LISBOA', 'SINTRA', 'SINTRA', 'VILA NOVA DE GAIA', 'PORTO', 'CASCAIS', 'LOURES', 'BRAGA', 'MATOSINHOS', 'AMADORA', 'OEIRAS', 'GONDOMAR', 'SEIXAL']    # Big boys only

            constParams = {'beta': args.betaConst, 'gamma': args.gammaConst, 'delta': args.deltaConst}

            # wandb.init(project="intake-sird-prediction")
            # wandb.config.update(args)
            # rootDir = wandb.run.dir

            rootDir = '../data'

            for forecastStartDate in simDates:

                outputDir = ol.getDir(rootDir, (forecastStartDate+timedelta(days=1)).strftime("%Y-%m-%d"))

                predictionDf = makePredictions(regionalDf, regions, timePeriod, forecastStartDate, args.forecastLength,
                                               args.paramPredictMethod, args.movAvgInterval, args.constantParams,
                                               args.pseudoCounts, toDB=False,
                                               outputPlots=True, outputDir=outputDir, constantParams=constParams, paramPredictInterval=args.paramPredictInterval)

                pl.save(predictionDf, 'csv', join(outputDir, f'predictions'))
        else:
            pass
