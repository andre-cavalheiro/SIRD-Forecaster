from os.path import join
import math

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import wandb

figSize = (55, 55)
DPI = 120

def smallMultiplesForecastV2(historicDf, forecastDf, historicPeriod, forecastPeriod, targetRegions, outputDir,
                             outputName, maPeriod=None):
    plotsPerRow = 5
    numPlots = len(targetRegions)
    numRows = math.ceil((numPlots / plotsPerRow))

    # Initialize the figure
    f = plt.figure(figsize=figSize)
    plt.style.use('seaborn-darkgrid')

    num = 0
    for region in targetRegions:
        num += 1

        f.add_subplot(numRows, plotsPerRow, num)
        if maPeriod is not None:
            historicDf.loc[region].loc[historicPeriod[0]:historicPeriod[-1]]['new-cases'].rolling(maPeriod,
                                                                                                  min_periods=1).mean().plot(
                marker='', color='grey', linewidth=2.4, alpha=0.9, ax=plt.gca())
        else:
            historicDf.loc[region].loc[historicPeriod[0]:historicPeriod[-1]]['new-cases'].plot(marker='', color='grey',
                                                                                               linewidth=2.4, alpha=0.9,
                                                                                               ax=plt.gca())

        forecastDf.loc[slice(forecastPeriod[0], forecastPeriod[-1])].xs(region, level=1)['predictedCases'].plot(
            marker='', color='black', linewidth=2.4, alpha=0.9, ax=plt.gca())
        r = 1

        '''
        if historicDf.index.isin([forecastPeriod[-1]], level=1).any():
            if maPeriod is not None:
                historicDf.loc[region].loc[forecastPeriod[1]:historicPeriod[-1]]['new-cases'].rolling(maPeriod, min_periods=1).mean().plot(marker='', color='red', linewidth=2.4, alpha=0.9, ax=plt.gca())
            else:
                historicDf.loc[region].loc[forecastPeriod[1]:historicPeriod[-1]]['new-cases'].plot(marker='', color='red', linewidth=2.4, alpha=0.9, ax=plt.gca())
        '''

        # Not ticks everywhere
        if (num - 1) < numPlots - plotsPerRow:
            plt.tick_params(labelbottom='off')
        if (num - 1) % plotsPerRow != 0:
            plt.tick_params(labelleft='off')

        # Add title
        plt.title(region, loc='left', fontsize=12, fontweight=0)

        plt.tight_layout()

    plt.savefig(join(outputDir, f'{outputName}.png'), dpi=DPI)
    plt.close()


def smallMultiplesGenericWithProjected(dataPerRegion, targetRegions, attributes, forecastPeriod, maxP, outputDir,
                                       outputName, prints=True):
    plotsPerRow = 5
    numPlots = len(targetRegions)
    numRows = math.ceil((numPlots / plotsPerRow))

    # Initialize the figure
    fig = plt.figure(figsize=figSize)
    plt.style.use('seaborn-darkgrid')
    colors = ['maroon', 'navy', 'darkgreen', 'indigo']
    colors2 = ['salmon', 'dodgerblue', 'olivedrab', 'plum']

    num = 0
    for region in targetRegions:
        num += 1

        fig.add_subplot(numRows, plotsPerRow, num)

        dataToPlot = [dataPerRegion[region][att] for att in attributes]
        dataToPlot2 = [pd.Series(dataPerRegion[region][f'{att}-forecast'][1:], index=forecastPeriod) for att in attributes]

        indexes = [[i for i in range(len(t))] for t in dataToPlot]
        indexes2 = [[len(indexes[0]) + i for i in range(len(t))] for t in dataToPlot2]

        if num == 1:
            for i, (ts, id, att) in enumerate(zip(dataToPlot, indexes, attributes)):
                ts.plot(marker='', label=att, color=colors[i])
                # plt.plot(id, ts, marker='', label=att, color=colors[i])
            for i, (ts, id, att) in enumerate(zip(dataToPlot2, indexes2, attributes)):
                ts.plot(marker='', label=f'{att}-estimated', color=colors2[i])
                # plt.plot(id, ts, marker='', label=f'{att}-estimated', color=colors2[i])
        else:
            for i, (ts, id) in enumerate(zip(dataToPlot, indexes)):
                ts.plot(marker='', color=colors[i])
                # plt.plot(id, ts, marker='', color=colors[i])
            for i, (ts, id) in enumerate(zip(dataToPlot2, indexes2)):
                ts.plot(marker='', color=colors2[i])
                # plt.plot(id, ts, marker='', color=colors2[i])

        plt.ylim(0, maxP)

        # Not ticks everywhere
        if (num - 1) < numPlots - plotsPerRow:
            plt.tick_params(labelbottom='off')
        if (num - 1) % plotsPerRow != 0:
            plt.tick_params(labelleft='off')

        # Add title
        plt.title(region, loc='left', fontsize=12, fontweight=0)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:6], labels[:6], loc='lower right', prop={'size': 24})

    plt.tight_layout()

    plt.savefig(join(outputDir, f'{outputName}.png'), dpi=DPI)
    plt.close()


def forecastCorrelation(results, forecastPeriod, outputDir, outputName):

    for day in forecastPeriod:
        targets = results[day]['targets']
        predictions = results[day]['predictions']
        R2 = results[day]['R2']
        RMSE = results[day]['RMSE']

        localLimit = max(targets) if max(targets) > max(predictions) else max(predictions)

        plt.scatter(targets, predictions, alpha=0.5)

        plt.xlim(0, localLimit)
        plt.ylim(0, localLimit)

        plt.xlabel("True")
        plt.ylabel("Predict")

        plt.legend()
        plt.title(f'{day.strftime("%Y-%m-%d")} -- R2={round(R2, 4)} -- RMSE={round(RMSE, 4)}')
        plt.savefig(join(outputDir, f'{outputName}-{day.strftime("%Y-%m-%d")}.png'))
        plt.close()

    limit=0
    for day in forecastPeriod:
        targets = results[day]['targets']
        predictions = results[day]['predictions']

        limit = max(targets) if max(targets) > limit else limit
        limit = max(predictions) if max(predictions) > limit else limit

        plt.scatter(targets, predictions, alpha=0.5, label=day.strftime("%m/%d"))

    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.xlabel("True")
    plt.ylabel("Predict")
    plt.legend()
    plt.title(f'R2={round(results["fullPeriod"]["R2"], 4)} -- RMSE={round(results["fullPeriod"]["RMSE"], 4)}')
    plt.savefig(join(outputDir, f'{outputName}.png'))
    plt.close()


def logToWandb(results):
    forecastId = results['fullPeriod']['origDate'].strftime("%Y-%m-%d")

    for day, res in results.items():
        if day == 'fullPeriod':
            wandb.log({
                f'R2/{forecastId}': results['fullPeriod']["R2"],
                f'RMSE/{forecastId}': results['fullPeriod']["RMSE"],
            })
        else:
            wandb.log({
                f'{forecastId}/R2': res["R2"],
                f'{forecastId}/RMSE': res["RMSE"],
            })


def logToWandbByRegionType(results):
    forecastId = results["verySmall"]['fullPeriod']['origDate'].strftime("%Y-%m-%d")

    counter = 0
    for l, dt in results.items():
        for day, res in dt.items():
            if day == 'fullPeriod':
                wandb.log({
                    f'R2/{l}/{forecastId}': res["R2"],
                    f'RMSE/{l}/{forecastId}': res["RMSE"],
                })
            else:
                wandb.log({
                    f'{forecastId}/{l}/R2': res["R2"],
                    f'{forecastId}/{l}/RMSE': res["RMSE"],
                })


def smallMultiplesForecast(dataPerRegion, targetRegions, priorSize, predictionSize, outputDir, outputName,
                           forecastKey='total_cases_sird'):

    plotsPerRow = 5
    numPlots = len(targetRegions)
    numRows = math.ceil((numPlots / plotsPerRow))

    priorIndexes = [i for i in range(priorSize)]
    predictindexes = [i for i in range(priorSize, priorSize+predictionSize)]

    # Initialize the figure
    f = plt.figure(figsize=(30, 20))
    plt.style.use('seaborn-darkgrid')

    num = 0
    for region in targetRegions:
        num += 1

        f.add_subplot(numRows, plotsPerRow, num)

        prior = dataPerRegion[region]['priors']
        prediction = dataPerRegion[region][forecastKey]

        # len(priorIndexes), len(prior), len(predictindexes), len(prediction), len(target)

        plt.plot(priorIndexes, prior, marker='', color='grey', linewidth=2.4, alpha=0.9)
        plt.plot(predictindexes, prediction.tolist(), marker='', color='black', linewidth=2.4, alpha=0.9, label=region)

        if 'target' in dataPerRegion[region].keys():
            target = dataPerRegion[region]['target']
            targetIndexes = [i for i in range(priorSize, priorSize + len(target))]
            plt.plot(targetIndexes, target.tolist(), marker='', color='darkred', linewidth=2.4, alpha=0.9, label=region)

        # Same limits for everybody!
        plt.xlim(0, priorSize + predictionSize)
        # plt.ylim(-0.1, 4)

        # Not ticks everywhere
        if (num - 1) < numPlots - plotsPerRow:
            plt.tick_params(labelbottom='off')
        if (num - 1) % plotsPerRow != 0:
            plt.tick_params(labelleft='off')

        # Add title
        plt.title(region, loc='left', fontsize=12, fontweight=0)

        plt.tight_layout()

    plt.savefig(join(outputDir, f'{outputName}.png'), dpi=DPI)
    plt.close()


def smallMultiplesGeneric(dataPerRegion, targetRegions, attributes, outputDir, outputName, prints=True):
    plotsPerRow = 5
    numPlots = len(targetRegions)
    numRows = math.ceil((numPlots / plotsPerRow))

    # Initialize the figure
    fig = plt.figure(figsize=(50, 50))
    plt.style.use('seaborn-darkgrid')

    num = 0
    for region in targetRegions:
        num += 1

        fig.add_subplot(numRows, plotsPerRow, num)

        dataToPlot = [dataPerRegion[region][att] for att in attributes]
        indexes = [[i for i in range(len(t))] for t in dataToPlot]

        if num==1:
            for ts, id, att in zip(dataToPlot, indexes, attributes):
                plt.plot(id, ts, marker='', label=att)
        else:
            for ts, id in zip(dataToPlot, indexes):
                plt.plot(id, ts, marker='')

        # Not ticks everywhere
        if (num - 1) < numPlots - plotsPerRow:
            plt.tick_params(labelbottom='off')
        if (num - 1) % plotsPerRow != 0:
            plt.tick_params(labelleft='off')

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='lower right', prop={'size': 24})

        # Add title
        plt.title(region, loc='left', fontsize=12, fontweight=0)

        plt.tight_layout()

    plt.savefig(join(outputDir, f'{outputName}.png'), dpi=DPI)
    plt.close()


