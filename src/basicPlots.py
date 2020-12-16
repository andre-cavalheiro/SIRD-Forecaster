import logging
from datetime import datetime, timedelta

import pandas as pd

import libs.visualization as vz
from utils.loader import loadPredictions, loadRefinedData


def buildResults(historicDf, predictionDf, regions, timePeriod, forecastPeriod, movAvgInt=7):
    results = {}

    for r in regions:
        results[r] = {}

        results[r]['priors'] = historicDf.loc[r][timePeriod[0]: timePeriod[-1]]['new-cases'].rolling(movAvgInt, min_periods=1).mean().values
        results[r]['forecast'] = predictionDf.loc[r][forecastPeriod[0]: forecastPeriod[-1]]['predictedCases'].values

        if historicDf.index.isin([forecastPeriod[-1]], level=1).any():
            results[r]['target'] = historicDf.loc[r][forecastPeriod[0]: forecastPeriod[-1]]['new-cases'].rolling(movAvgInt, min_periods=1).mean().values

    return results


def basicPlot(historicDf, predictionDf, targetRegions, timePeriod, simulationStartDate, forecastStartDate,
              forecastLength, outputDir):

    # Adjust params

    logging.info(f'Assuming forecast starts at {forecastStartDate}')

    timePeriod = timePeriod[timePeriod.index(simulationStartDate):]
    logging.info(f'Historic data starting at {simulationStartDate}')

    forecastPeriod = [forecastStartDate + timedelta(days=x) for x in range(forecastLength)]

    # Plot forecast
    historicDf = historicDf.reset_index()
    historicDf = (historicDf.set_index(['Region', 'Day'], drop=True).sort_index())

    predictionDf = predictionDf.reset_index()
    predictionDf = (predictionDf.set_index(['Region', 'Day'], drop=True).sort_index())

    resultsReadyToPlot = buildResults(historicDf, predictionDf, targetRegions, timePeriod, forecastPeriod)

    logging.info('Plotting (in basic)...')
    vz.smallMultiplesForecast(resultsReadyToPlot, targetRegions, len(timePeriod), forecastLength, outputDir,
                              'forecast-new-cases', forecastKey='forecast')

if __name__ == '__main__':
    outputDir = '.'
    simulationStart = pd.to_datetime('2020-05-01', format='%Y-%m-%d')
    forecastStartDate = pd.to_datetime('2020-12-09', format='%Y-%m-%d')
    forecastLength = 15
    targetRegions = ['LISBOA', 'PORTO', 'BEJA', 'GUARDA', 'COVILHA', 'AMADORA', 'ODIVELAS', 'CASTELO BRANCO', 'FARO',
                    'CASCAIS', 'AVEIRO', 'MONTIJO', 'CALDAS DA RAINHA', 'ESPINHO', 'MAIA', 'SINTRA', 'PENAFIEL',
                    'VISEU', 'SETUBAL', 'BRAGA', 'BARCELOS', 'LAGOS', 'ALCOBACA', 'NAZARE', 'LOURES', 'TORRES VEDRAS']

    # Load
    predictionDf = loadPredictions()
    historicDf, _, timePeriod = loadRefinedData()

    basicPlot(historicDf, predictionDf, targetRegions, timePeriod, outputDir)
