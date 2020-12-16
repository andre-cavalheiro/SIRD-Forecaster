import time
import yaml
from numpy import array, hstack, asarray, mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load, dump
from os.path import join

def getConfiguration(configFile):
    with open(configFile, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def shouldEstablishConnection(row):
    connectionAB, connectionBA = False, False
    amountAB, amountBA = None, None
    if row['Nacionais AB'] != 0 or row['Estrangeiros AB'] != 0:
        connectionAB = True
        amountAB = (row['Nacionais AB'], row['Estrangeiros AB'])
    if row['Nacionais BA'] != 0 or row['Estrangeiros BA'] != 0:
        connectionBA = True
        amountBA = (row['Nacionais BA'], row['Estrangeiros BA'])
    return connectionAB, amountAB, connectionBA, amountBA


def dayHourCombinations(possibleDays, possibleHours):
    combinations = []
    for day in possibleDays:
        for hour in possibleHours:
            combinations.append({
                'day': day,
                'hour': hour
            })
    return combinations


def addToRegionDataset(df, regions):
    for r in regions:
        if not df['region'].isin([r]).any():
            df = df.append({
                'region': r,
                'regionID': df.shape[0]
            }, ignore_index=True)
    return df


def getRegionID(region, df):
    i = df[df['region'] == region]['regionID'].values[0]
    return i


def getDayFormated(day, hour):
    d = time.strptime(day, '%Y-%m-%d %H:%M:%S')
    h = time.strptime(hour, '%Y-%m-%d %H:%M:%S')

    dayFormated = '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(d.tm_year, d.tm_mon, d.tm_mday, h.tm_hour, h.tm_min, h.tm_sec)
    return dayFormated


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, predictionSize):
    X, y = list(), list()
    # for i in range(len(sequences)):
    for i in range(0, len(sequences), n_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        endInd = end_ix-1
        # check if we are beyond the dataset
        if end_ix > len(sequences) or endInd+predictionSize>len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1].tolist(), sequences[endInd:endInd+predictionSize, -1].tolist()
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)


def movingAvg(series, interval, numDecimalPoint=3):
    '''
    :param series: np.array
    :param interval: int
    :return: np.array
    '''
    # mean = [np.mean(series[x:x + interval]) for x in range(len(series) - interval + 1)]   # -> normie way
    simpleMA = pd.Series(series).rolling(interval).mean()
    simpleMA = np.round(simpleMA, decimals=numDecimalPoint)
    return simpleMA.values

def expMovingAvg(series, interval, numDecimalPoint=3):
    '''
    :param series: np.array
    :param interval: int
    :return: np.array
    '''
    expMA = pd.Series(series).ewm(span=interval).mean()
    expMA = np.round(expMA, decimals=numDecimalPoint)
    return expMA.values


def smoothMAcum(series, interval, numDecimalPoint=3): # Moving average by cumsum, scale = window size in m
    # src: https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    cumsum = np.cumsum(np.insert(series, 0, 0))
    smoothed = (cumsum[interval:] - cumsum[:-interval]) / interval
    smoothed = np.round(smoothed, decimals=numDecimalPoint)
    return smoothed

# USE THIS - AT LEAST FOR KNOW! KEEPS DATA SIZE INTACT
def smoothMAconv(series, interval, numDecimalPoint=3): # Moving average by numpy convolution
    # This one makes sure the input and output are of the same size
    # src: https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    y_padded = np.pad(series, (interval // 2, interval - 1 - interval // 2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((interval,)) / interval, mode='valid')
    y_smooth = np.round(y_smooth, decimals=numDecimalPoint)

    #plt.plot(series)
    #plt.plot(y_smooth)
    #plt.show()
    return y_smooth


def loadXY(dir, type):
    with open(join(dir, f"X-{type}.pkl"), "rb") as f:
        X = load(f)
    with open(join(dir, f"y-{type}.pkl"), "rb") as f:
        y = load(f)
    return X, y


def dumpXY(X, y, dir, type):
    with open(join(dir, f"X-{type}.pkl"), "wb") as f:
        X = dump(X, f)
    with open(join(dir, f"y-{type}.pkl"), "wb") as f:
        y = dump(y, f)
    return X, y