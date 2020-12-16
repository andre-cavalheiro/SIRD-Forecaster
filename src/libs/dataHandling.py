from os.path import join

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle

from libs import jsonLib as jl
import libs.pandasLib as pl


def loadDataset(location, name, format, cutBy=None, **kwargs):
    fullPath = join(location, '{}.{}'.format(name, format))
    if format == 'csv': #  or format == 'pickle'
        data = pl.load(fullPath, format=format, **kwargs)
    elif format == 'json':
        data = pl.load(fullPath, format=format, **kwargs)
    elif format == 'jsonDict':      # json as dictionary
        fullPath = join(location, '{}.{}'.format(name, 'json'))
        data = jl.load(fullPath, **kwargs)
    elif format == 'gpickle':   # graph
        data = nx.read_gpickle(fullPath, **kwargs)
    elif format == 'npy':
        data = np.load(fullPath, **kwargs)
    elif format == 'npz':   # Sparse
        data = load_npz(fullPath, **kwargs)
    elif format == 'pickle':
        with open(fullPath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise Exception('Unknown file format: {}'.format(format))

    if cutBy is not None:
        print('Cutting dataset {} by {}'.format(location, cutBy))
        data, _ = dropDatasetBy(data, cutBy, format)
    return data


def saveDataset(location, name, format, df, **kwargs):
    # fixme - MAYBE if the location does not exist -> create it. Not sure. Think about it.
    fullPath = join(location, '{}.{}'.format(name, format))
    if format == 'csv': #  or format == 'pickle'
        df.to_csv(fullPath, **kwargs)
    elif format == 'gpickle':   # graph
        nx.write_gpickle(df, fullPath, **kwargs)
    elif format == 'npy':
        np.save(fullPath, df, **kwargs)
    elif format == 'npz':   # Sparse
        np.save(fullPath, df, **kwargs)
    elif format == 'pickle':
        with open(fullPath, 'wb') as f:
            pickle.dump(df, f)
    else:
        raise Exception('Unkown format')


def exists(location, name, format, verification=None):
    fullName = '{}.{}'.format(name, format)
    file = fileExists(location, fullName)
    if verification:
        if verification(file):
            return file
        else:
            raise Exception('Failed file verification - file location: {}'.format(location))

    return file


def loadDatasetsIntoDictionary(datasetNames, datasetConfigs):
    dfs = {}
    for inputName in datasetNames:
        location, name, format, remainingParams = unpackDictionary(datasetConfigs[inputName],
                                                                   ['location', 'name', 'format'], raiseExeptions=True)
        df = loadDataset(location, name, format, **remainingParams)
        dfs[inputName] = df

    return dfs

def dropDatasetBy(df, percentage, format):
    if format == 'json':
        data, droppedData = jl.dropByPercentage(df, percentage)
        return data, droppedData

    elif format=='csv':
        data, droppedData = pl.dropByPercentage(df, percentage)
        return data, droppedData



