import pandas as pd


def load(fileName, format, **kwargs):
    fullPath = f'{fileName}.{format}'
    if format == 'csv':
        return pd.read_csv(fullPath, **kwargs)
    elif format == 'pickle':
        return pd.read_pickle(fullPath, **kwargs)
    elif format == 'xlsx':
        return pd.read_excel(fullPath, **kwargs)
    else:
        raise Exception('Unknown load type {}'.format(format))


def save(df, format, fileName, **kwargs):
    fullPath = f'{fileName}.{format}'
    if format == 'csv':
        return df.to_csv(fullPath, **kwargs)
    elif format == 'pickle':
        return df.to_pickle(fullPath, **kwargs)
    else:
        raise Exception('Unknown load type {}'.format(format))


def dropByPercentage(df, percentage):
    '''
        Drop {percentage}% of the dataset.
    :param df: pandas.DataFrame
    :param percentage: float
    :return:
    '''
    stopIndex = int(df.shape[0] * (1-percentage))
    droppedDf = df.iloc[stopIndex:, :]
    df = df.iloc[:stopIndex, :]
    return df, droppedDf


def uniqueValuesForAttr(df, attributeName):
    '''
    Get unique values for specific attribute in the dataset
    :param df: pandas.DataFrame
    :param attributeName: String
    :return: numpy.array
    '''
    attributeList = df[attributeName]
    uniqueValList = attributeList.unique()
    return uniqueValList


def dropColumns(df, columns, **kwargs):
    droppedCols = df[columns]
    df = df.drop(labels=columns, axis=1, **kwargs)
    return df, droppedCols


def dropRows(df, rows, **kwargs):
    droppedRows = df.ix[rows]
    df = df.drop(labels=rows, axis=0, **kwargs)
    return df, droppedRows


def concat(*dfs, reIndex = True, **kwargs):
    frames = [f for f in dfs]
    df = pd.concat(frames, **kwargs)    # sort=True
    if reIndex:
        df = df.reset_index(drop=True)
    return df


