from os.path import join

import wandb
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.colors import LogNorm

def numDays(month):
    if month=='Setembro':
        return 30
    elif month=='Outubro':
        return 31
    elif month=='Novembro':
        return 30
    elif month=='Dezembro':
        return 31
    elif month=='Janeiro':
        return 30
    elif month=='Fevereiro':
        return 28
    else:
        raise Exception('Unknown month')


def monthNumber(month):
    if month=='Setembro':
        return '09'
    elif month=='Outubro':
        return '10'
    elif month=='Novembro':
        return '11'
    elif month=='Dezembro':
        return '12'
    elif month=='Janeiro':
        return '01'
    elif month=='Fevereiro':
        return '02'
    else:
        raise Exception('Unknown month')


def getYear(month):
    if month in ['Setembro', 'Outubro', 'Novembro', 'Dezembro']:
        return '2020'
    elif month in ['Janeiro', 'Fevereiro']:
        return '2021'
    else:
        raise Exception('Unknown month')


def loadEverything(rootDir, subDir='Previsto', months=['Setembro', 'Outubro', 'Novembro', 'Dezembro', 'Janeiro', 'Fevereiro'], modelExtension='IQD', fileExtension='asc'):
    output = []
    for month in months:
        monthIndex = monthNumber(month)
        year = getYear(month)
        for dayIdx in range(1, numDays(month)+1):
            if dayIdx < 10:
                dayAsStr = '0' + str(dayIdx)
            else:
                dayAsStr = str(dayIdx)

            day = '_'.join([year, monthIndex, dayAsStr])
            fileName = f'{day}_{modelExtension}.{fileExtension}'

            #print(fileName)

            filePath = join(rootDir, month, subDir, fileName)
            M = np.genfromtxt(filePath, dtype=None, skip_header = 6)
            output.append(M)
    return output


def loadM(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def saveM(m, name):
    with open(name, 'wb') as f:
        pickle.dump(m, f)


def convertToMPerDay(M, targetDimentionPerDay=(288, 141), numDays=116):
    output = []

    for dayIdx in range(M.shape[1]):
        dayData = M[:, dayIdx]
        dayData2D = np.reshape(dayData, (-1, targetDimentionPerDay[1]))
        # assert(dayData2D.shape == targetDimentionPerDay)
        dayData2D = np.flipud(dayData2D)     # flip it otherwise heatmap comes out backwards
        output.append(dayData2D)

    # assert(len(output) == numDays)
    return output


def convertToMPerDay(M, targetDimentionPerDay=(288, 141)):
    output = []

    for dayIdx in range(M.shape[0]):
        dayData = M[dayIdx, :]
        dayData2D = np.reshape(dayData, (-1, targetDimentionPerDay[1]))
        assert(dayData2D.shape == targetDimentionPerDay)
        dayData2D = np.flipud(dayData2D)     # flip it otherwise heatmap comes out backwards
        output.append(dayData2D)

    return output


def calcErrMPerDot(targetList, predictionList, rmse=False, saveAsWeGo=False):
    output = []

    for dayIdx, (t, p) in enumerate(zip(targetList, predictionList)):
        #print(dayIdx)

        errM = np.zeros(t.shape)
        dotsIgnored = 0
        for rowIdx in range(t.shape[0]):
            for colIdx in range(t.shape[1]):

                if t[rowIdx, colIdx] == -999.250 or p[rowIdx, colIdx] == -999.250:
                    dotsIgnored +=1
                    errM[rowIdx, colIdx] = None
                    continue

                errM[rowIdx, colIdx] = mean_squared_error(
                    np.array([t[rowIdx, colIdx]]),
                    np.array([p[rowIdx, colIdx]]),
                    squared=not rmse
                )
        # print(f'Ignored {dotsIgnored} dots')
        output.append(errM)

        if saveAsWeGo is True:
            k = 'RMSE' if rmse is True else 'MSE'
            saveM(errM, f'{k}-Matrix-{dayIdx}.pkl')

    return output


def calcPercentErrMPerDot(targetList, predictionList, saveAsWeGo=False):
    output = []

    for dayIdx, (t, p) in enumerate(zip(targetList, predictionList)):
        print(dayIdx)

        errM = np.zeros(t.shape)
        dotsIgnored = 0
        for rowIdx in range(t.shape[0]):
            for colIdx in range(t.shape[1]):

                if t[rowIdx, colIdx] == -999.250 or p[rowIdx, colIdx] == -999.250:
                    dotsIgnored +=1
                    errM[rowIdx, colIdx] = None
                    continue

                errM[rowIdx, colIdx] = 2*abs(p[rowIdx, colIdx]-t[rowIdx, colIdx])/(abs(t[rowIdx, colIdx])+abs(p[rowIdx, colIdx]))

        # print(f'Ignored {dotsIgnored} dots')
        output.append(errM)

        if saveAsWeGo is True:
            saveM(errM, f'Percentage-Matrix-{dayIdx}.pkl')

    return output


def uploadErrPerDay(errorList, key):
    dailyErrors = []
    for dayIdx, errM in enumerate(errorList):
        dailyErrPerValidDot = [x for row in errM for x in row if not np.isnan(x)]
        # print(f'Valid dots: {len(dailyErrPerValidDot)} VS full dots: {errM.shape[0] * errM.shape[1]}')
        dailyErr = sum(dailyErrPerValidDot)/len(dailyErrPerValidDot)
        dailyErrors.append(dailyErr)
        # print(f'{dayIdx} - {key} - {dailyErr}')
        wandb.log({
            f'Daily/{key}': dailyErr,
            'day': dayIdx,
        })
    return dailyErrors


def uploadErrPerWeek(errorList, key):
    weekIdx = 0
    for it in range(0, len(errorList),  7):

        errWeek = errorList[it: it+6] if it+6 < len(errorList) else errorList[it:]

        err = [x for dayM in errWeek for row in dayM for x in row if not np.isnan(x)]

        wandb.log({
            f'Weekly/{key}': sum(err)/len(err),
            'week': weekIdx
        })
        weekIdx += 1


def boxPlotPerDay(errorList, key):
    plotDf = None
    for dayIdx, errM in enumerate(errorList):
        dailyErrPerValidDot = [x for row in errM for x in row if not np.isnan(x)]
        auxDf = pd.Series(dailyErrPerValidDot).to_frame(name=key)
        auxDf['Day'] = dayIdx+6
        plotDf = auxDf if plotDf is None else pd.concat([plotDf, auxDf], ignore_index=False)

    boxPlot(plotDf, key, 'Day', 'daily')
    # violinPlot(plotDf, key, 'Day', 'daily')


def boxPlotPerWeek(errorList, key):
    weekIdx = 0
    plotDf = None

    for it in range(0, len(errorList), 7):
        errWeek = errorList[it: it + 6] if it + 6 < len(errorList) else errorList[it:]
        err = [x for dayM in errWeek for row in dayM for x in row if not np.isnan(x)]

        auxDf = pd.Series(err).to_frame(name=key)
        auxDf['Week'] = weekIdx
        plotDf = auxDf if plotDf is None else pd.concat([plotDf, auxDf], ignore_index=False)

        weekIdx += 1

    boxPlot(plotDf, key, 'Week', 'weekly')
    # violinPlot(plotDf, key, 'Week', 'weekly')


def boxPlot(plotDf, key, xKey, timePeriod):

    # Make plot
    fig, ax = plt.subplots(figsize=(35, 25))
    g = sns.boxplot(ax=ax, data=plotDf, x=xKey, y=key, palette="Set2", showfliers=True, showmeans=True)

    # Fix axis
    g.set_xlabel(xKey, fontsize=20)
    g.set_ylabel(key, fontsize=20)
    g.set_yticklabels(g.get_yticks(), size=16)
    g.set_xticklabels(g.get_xticks(), size=14)

    # Upload it as a stand alone image as well
    plt.savefig(join(wandb.run.dir, f'BoxPlot-{timePeriod}-{key}.png'), dpi=150)

    # Upload it as wandb plot
    # wandb.log({f'Boxplot/{timePeriod}/{key}': fig})

    plt.close()


def heatMap(errorList, key):
    weekIdx = 1

    for it in range(0, len(errorList), 7):

        M = np.mean(errorList[it: it + 6], axis=0) if it+6 < len(errorList) else np.mean(errorList[it:], axis=0)

        maskedArr = np.ma.masked_invalid(M)

        fig = plt.figure()
        cmap = plt.cm.Reds
        cmap.set_bad(color='white')

        plt.imshow(maskedArr, cmap=cmap, norm=LogNorm(vmin=0, vmax=2))
        plt.axis('off')
        plt.colorbar()

        wandb.log({f'Heatmap ({key}) Week:{weekIdx}': plt})

        plt.close()

        weekIdx += 1

def heatMap(errorList, key):
    weekIdx = 1

    for it in range(0, len(errorList), 7):

        M = np.mean(errorList[it: it + 6], axis=0) if it+6 < len(errorList) else np.mean(errorList[it:], axis=0)

        maskedArr = np.ma.masked_invalid(M)

        fig = plt.figure()
        cmap = plt.cm.Reds
        cmap.set_bad(color='white')

        plt.imshow(maskedArr, cmap=cmap, norm=LogNorm(vmin=0, vmax=2))
        plt.axis('off')
        plt.colorbar()

        wandb.log({f'Heatmap ({key}) Week:{weekIdx}': plt})

        plt.close()

        weekIdx += 1


def heatMapMultipleLOG(errorListPerModel, key):
    models = list(errorListPerModel.keys())
    numDays = len(errorListPerModel['SIRD'])

    weekIdx = 1
    for it in range(0, numDays, 7):

        MPerModel = {}
        currMax = 0

        for model in models:
            errorList = errorListPerModel[model]
            M = np.mean(errorList[it: it + 6], axis=0) if it+6 < len(errorList) else np.mean(errorList[it:], axis=0)
            maskedArr = np.ma.masked_invalid(M)
            MPerModel[model] = maskedArr

            maxErr = np.max(maskedArr)
            currMax = maxErr if maxErr > currMax else currMax

        cmap = plt.cm.Reds
        cmap.set_bad(color='white')
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for ax, model in zip(axes.flat, models):
            heatMapData = MPerModel[model]
            if model == 'SIRD':
                im = ax.imshow(heatMapData, cmap=cmap, norm=LogNorm(vmin=1, vmax=currMax))
            else:
                im = ax.imshow(np.flipud(heatMapData), cmap=cmap, norm=LogNorm(vmin=1, vmax=currMax))

            ax.title.set_text(model)
            ax.axis('off')

        fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(join(rootDir, f'Heatmap ({key} - week {weekIdx}.png'), dpi=100)
        plt.close()

        weekIdx += 1


def heatMapMultiple(errorListPerModel, key):
    models = list(errorListPerModel.keys())
    numDays = len(errorListPerModel['SIRD'])

    weekIdx = 1
    for it in range(0, numDays, 7):

        MPerModel = {}
        currMax = 0

        for model in models:
            errorList = errorListPerModel[model]
            M = np.mean(errorList[it: it + 6], axis=0) if it+6 < len(errorList) else np.mean(errorList[it:], axis=0)
            maskedArr = np.ma.masked_invalid(M)
            MPerModel[model] = maskedArr

            maxErr = np.max(maskedArr)
            currMax = maxErr if maxErr > currMax else currMax

        cmap = plt.cm.Reds
        cmap.set_bad(color='white')
        fig, axes = plt.subplots(nrows=2, ncols=2)
        for ax, model in zip(axes.flat, models):
            heatMapData = MPerModel[model]
            if model == 'SIRD':
                im = ax.imshow(heatMapData, cmap=cmap, vmin=0, vmax=2)
            else:
                im = ax.imshow(np.flipud(heatMapData), cmap=cmap,vmin=0, vmax=2)

            ax.title.set_text(model)
            ax.axis('off')

        fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)
        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(join(rootDir, f'Heatmap ({key} - week {weekIdx}.png'), dpi=100)
        plt.close()

        weekIdx += 1


def getVar(errorListPerModel, key):

    for model, errorList in errorListPerModel.items():
        total = []

        for dayIdx, errM in enumerate(errorList):
            dailyErrPerValidDot = [x for row in errM for x in row if not np.isnan(x)]
            dailyErr = sum(dailyErrPerValidDot)/len(dailyErrPerValidDot)
            total.append(dailyErr)
        print(f'{model} ({key}) -> {np.var(total)}')



if __name__ == '__main__':

    rootDir = '../data/paperData'
    initalProcessing, wandbLot, boxPlotTogether = False, False, True
    modelExtension = 'medianmodel'

    if initalProcessing is True:

        targetList = loadEverything(rootDir, subDir='Real', modelExtension=modelExtension)
        predictionList = loadEverything(rootDir, subDir='Previsto', modelExtension=modelExtension)

        '''
        targetM = np.genfromtxt(join(rootDir, 'simulationsMatrix.asc'), dtype=None)
        targetList = convertToMPerDay(targetM)
        predictionM = np.genfromtxt(join(rootDir, 'predictionsMatrix.asc'), dtype=None)
        predictionList = convertToMPerDay(predictionM)
        '''

        # MSE
        #errorListMSE = calcErrMPerDot(targetList, predictionList, rmse=False, saveAsWeGo=False)
        #saveM(errorListMSE, join(rootDir, f'MSE-Matrix-Full-{modelExtension}.pkl'))

        # RMSE
        errorListRMSE = calcErrMPerDot(targetList, predictionList, rmse=True, saveAsWeGo=False)
        saveM(errorListRMSE, join(rootDir, f'RMSE-Matrix-Full-{modelExtension}.pkl'))

        # Percentual error
        errorListPercentual = calcPercentErrMPerDot(targetList, predictionList, saveAsWeGo=False)
        saveM(errorListPercentual, join(rootDir, f'Percentual-Matrix-Full-{modelExtension}.pkl'))
    else:
        #errorListMSE = loadM(join(rootDir, f'MSE-Matrix-Full-{modelExtension}.pkl'))
        errorListRMSE = loadM(join(rootDir, f'RMSE-Matrix-Full-{modelExtension}.pkl'))
        errorListPercentual = loadM(join(rootDir, f'Percentual-Matrix-Full-{modelExtension}.pkl'))

    if wandbLot is True:
        wandb.init(project='intake-predict-eval', entity='intake')

        print('Logging')

        # Daily stats
        #boxPlotPerDay(errorListRMSE, key='RMSE')
        #boxPlotPerDay(errorListPercentual, key='Percentual')

        #errorsMSE = uploadErrPerDay(errorListMSE, key='MSE')
        errorsRMSE = uploadErrPerDay(errorListRMSE, key='RMSE')
        errorsPercentual = uploadErrPerDay(errorListPercentual, key='Percentual')

        #globalMetric('MSE', errorsMSE)
        globalMetric('RMSE', errorsRMSE)
        globalMetric('Percentual', errorsPercentual)

        print('>>> Logging')

        # Weekly stats
        #boxPlotPerWeek(errorListRMSE, key='RMSE')
        #boxPlotPerWeek(errorListPercentual, key='Percentual')

        #uploadErrPerWeek(errorListMSE, key='MSE')
        uploadErrPerWeek(errorListRMSE, key='RMSE')
        uploadErrPerWeek(errorListPercentual, key=' Percentual')
        
        print('>>> Logging')

        heatMap(errorListRMSE, key='RMSE')
        heatMap(errorListPercentual, key=' Percentual')

    if boxPlotTogether is True:

        createBoxDf = False

        if createBoxDf:
            # Percentual Error
            rmseErrFiles = {
                'ARMA': 'losses_rmse_arma.asc',
                'STConvS2S': 'losses_rmse_stconvs2s.asc',
                'VAR': 'losses_rmse_var.asc'
            }

            rmseErrPerModel = {
                'SIRD': errorListRMSE
            }

            print('>>> Loading')
            for model, f in rmseErrFiles.items():
                print(model)
                M = np.load(join(rootDir, 'losses', f))
                MList = convertToMPerDay(M)
                rmseErrPerModel[model] = MList

            print('>>> Calculating')
            plotDf = None
            for model, errorList in rmseErrPerModel.items():
                print(model)
                for dayIdx, errM in enumerate(errorList):
                    dailyErrPerValidDot = [x for row in errM for x in row if not np.isnan(x)]
                    auxDf = pd.Series(dailyErrPerValidDot).to_frame(name='RMSE')
                    auxDf['Day'] = dayIdx
                    auxDf['Model'] = model
                    plotDf = auxDf if plotDf is None else pd.concat([plotDf, auxDf], ignore_index=False)

            rmseBoxDf = plotDf

            # Save
            rmseBoxDf.to_csv(join(rootDir, f'RMSE-boxplotDf.csv'))
            saveM(rmseErrPerModel, join(rootDir, f'RMSE-data.pikcle'))   # Save as pickle

        else:
            rmseBoxDf = pd.read_csv((join(rootDir, f'RMSE-boxplotDf.csv')))
            rmseErrPerModel = loadM(join(rootDir, f'RMSE-data.pikcle'))   # Save as pickle
            getVar(rmseErrPerModel, 'RMSE')


        '''
        sns.set(font_scale=1.5)
        g = sns.FacetGrid(rmseBoxDf, col='Model', col_wrap=1, margin_titles=True)
        g.fig.set_size_inches(36, 15)
        g.map(sns.boxplot, "Day", "RMSE", showfliers=False, showmeans=True)
        g.set(xticks=[i for i in range(1, len(errorListRMSE)) if i%5==0])
        plt.savefig(join(rootDir, f'BoxPlot-RMSE.png'), dpi=100)
        plt.close()
        sns.set(font_scale=1)

        # Heatmaps
        heatMapMultipleLOG(rmseErrPerModel, 'RMSE')
        '''
        # ----------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------

        # SMAPE Error
        if createBoxDf:

            percentalErrFiles = {
                'ARMA': 'losses_percentual_arma.asc',
                'STConvS2S': 'losses_percentual_stconvs2s.asc',
                'VAR': 'losses_percentual_var.asc'
            }

            percentualErrPerModel = {
                'SIRD': errorListPercentual
            }

            print('>>> Loading')
            for model, f in percentalErrFiles.items():
                print(model)
                M = np.load(join(rootDir, 'losses', f))
                MList = convertToMPerDay(M)
                percentualErrPerModel[model] = MList

            print('>>> Calculating')
            plotDf = None
            for model, errorList in percentualErrPerModel.items():
                print(model)
                for dayIdx, errM in enumerate(errorList):
                    dailyErrPerValidDot = [x for row in errM for x in row if not np.isnan(x)]
                    auxDf = pd.Series(dailyErrPerValidDot).to_frame(name='SMAPE')
                    auxDf['Day'] = dayIdx
                    auxDf['Model'] = model
                    plotDf = auxDf if plotDf is None else pd.concat([plotDf, auxDf], ignore_index=False)

            percentualBoxDf = plotDf

            # Save
            percentualBoxDf.to_csv(join(rootDir, f'Percentual-boxplotDf.csv'))
            saveM(percentualErrPerModel, join(rootDir, f'Percentual-data.pikcle'))   # Save as pickle

        else:
            percentualBoxDf = pd.read_csv((join(rootDir, f'Percentual-boxplotDf.csv')))
            percentualErrPerModel = loadM(join(rootDir, f'Percentual-data.pikcle'))   # Save as pickle
            getVar(percentualErrPerModel, 'Percentual')
        exit()
        # Heatmaps
        heatMapMultiple(percentualErrPerModel, 'SMAPE')

        sns.set(font_scale=1.5)
        g = sns.FacetGrid(percentualBoxDf, col='Model', col_wrap=1, margin_titles=True)
        g.fig.set_size_inches(36, 15)
        g.map(sns.boxplot, "Day", "SMAPE", showfliers=False, showmeans=True)
        g.set(xticks=[i for i in range(1, len(errorListRMSE)) if i%5==0])
        plt.savefig(join(rootDir, f'BoxPlot-SMAPE.png'), dpi=100)
        plt.close()
        sns.set(font_scale=1)  # crazy big



