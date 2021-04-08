import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
from unidecode import unidecode
from os.path import join
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

import libs.pandasLib as pl
from libs.yamlLib import loadConfig


def loadNewDf(previousLatestDate, locationAttr, dateAttr, insertionAttr):

    # Load db authentication data
    auth = loadConfig()

    # Set up DB client
    engine = create_engine(
        f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['originalDB']}"
    )
    logging.info('Connection established!')

    # Find latest insertion date
    query = 'SELECT input_date FROM dgs_new ORDER BY input_date DESC LIMIT 1;'
    latestInputDate = pd.read_sql(query, con=engine)
    latestInputDate = latestInputDate.loc[0]['input_date']

    # Get dataset
    query = f"SELECT {locationAttr}, {dateAttr} FROM dgs_new " \
            f"WHERE {insertionAttr}=\'{latestInputDate}\' AND {dateAttr}>\'{previousLatestDate}\';"
    newDf = pd.read_sql(query, con=engine)

    engine.dispose()

    # Make sure we have no nans
    previousShape = newDf.shape[0]
    newDf = newDf.replace('NULL', np.nan).dropna(how='any')
    newShape = newDf.shape[0]
    if previousShape != newShape:
        logging.warning(f'Dropped {previousShape-newShape} rows due to NANs')

    # Transform location strings to unicode, also standardize region names (all upper case)
    # newDf['originalReg'] = newDf[locationAttr]
    newDf[locationAttr] = newDf[locationAttr].apply(lambda x: unidecode(x)).str.upper()

    # Save region name mapping - should only when run when processing all the data from the begging.
    '''
    uniqueOrigRegs, uniqueRegs = newDf['originalReg'].unique(), newDf[locationAttr].unique()
    regMapping = {k: v for k, v in zip(uniqueRegs, uniqueOrigRegs)}
    with open('../data/regionMapping.json', 'w') as outfile:
        json.dump(regMapping, outfile)
    '''

    # Format dates
    newDf[dateAttr] = pd.to_datetime(newDf[dateAttr]).dt.normalize()

    # Set time interval we're processing
    currLatestDate = newDf[dateAttr].max()
    delta = (currLatestDate - previousLatestDate).days
    coveredPeriod = [previousLatestDate + timedelta(days=i) for i in range(1, delta+1)]
    assert(coveredPeriod[-1] == currLatestDate)

    return newDf, coveredPeriod


def loadOldLatestDf():
    # Load db authentication data
    auth = loadConfig()

    # Set up DB client
    engine = create_engine(
        f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
    )
    logging.info('Connection established!')

    # Find latest date
    query = 'SELECT Day FROM cases ORDER BY Day DESC LIMIT 1;'
    latestDate = pd.read_sql(query, con=engine)
    latestDate = pd.to_datetime(latestDate.loc[0]['Day'], format='%Y-%m-%d')

    # Get dataset
    query = f"SELECT * FROM cases WHERE Day=\'{latestDate}\';"
    df = pd.read_sql(query, con=engine)
    df['Region'] = df["Region"].apply(lambda x: unidecode(x)).str.upper()       # SÃO JOÃO DA MADEIRA
    df = df.set_index('Region')
    engine.dispose()
    return df, latestDate


def pullNationalData(regions, timePeriod):

    dataDir = '../data'

    openSourceDf = pl.load(join(dataDir, 'openSource'), 'csv')

    openSourceDf = openSourceDf.set_index("data", drop=True)
    openSourceDf.index = pd.to_datetime(openSourceDf.index, format='%d-%m-%Y').normalize()

    openSourceDf["novos_obitos"] = openSourceDf["obitos"].diff().fillna(0)
    openSourceDf["novos_confirmados"] = openSourceDf["confirmados"].diff().fillna(0)

    latestDate = openSourceDf.index.max()

    for d in timePeriod:
        if d not in openSourceDf.index:
            newRow = pd.Series({k: 0 for k in openSourceDf.columns})
            newRow.name = d
            openSourceDf = openSourceDf.append(newRow)
            r=1
            logging.warning(f'National Df does not contain the day: {d} - filling with zeros')

    return openSourceDf, latestDate


def loadRefinedData(locationAttr='Region', dateAttr='Day', fromOS=False, inputDir='../data'):
    if fromOS is True:
        df = pl.load(join(inputDir, 'casesPerRegion'), format='csv')
        df[dateAttr] = pd.to_datetime(df[dateAttr]).dt.normalize()
        coveredRegions = pl.uniqueValuesForAttr(df, locationAttr).tolist()
        coveredDays = pd.Index(sorted(df[dateAttr].unique())).tolist()
        df = (df.set_index([locationAttr, dateAttr]).sort_index())
        df = df.drop_duplicates()
        return df, coveredRegions, coveredDays

    # Load db authentication data
    auth = loadConfig()

    # Set up DB client
    engine = create_engine(
        f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
    )
    logging.info('Connection established!')

    # Get dataset
    query = 'SELECT * FROM cases;'
    df = pd.read_sql(query, con=engine)

    # Transform location strings to unicode, also standardize region names (all upper case)
    df[locationAttr] = df[locationAttr].str.upper()

    # Format dates
    df[dateAttr] = pd.to_datetime(df[dateAttr]).dt.normalize()

    # Get unique days
    coveredDays = pd.Index(sorted(df[dateAttr].unique())).tolist()

    # Set index
    df = (df.set_index([locationAttr, dateAttr]).sort_index())

    engine.dispose()

    # pl.save(df, 'csv', join('../data', f'casesPerRegion'))

    return df, coveredDays


def loadPopulation():
    df = pl.load(join('../data/', 'PORDATA_PopulacaoResidente'), 'xlsx', header=11)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')     # Specific to this weird dataset
    df = df.rename(columns={'Anos': 'Region'})
    df = df[df['Region'].notna()]
    df['Region'] = df["Region"].apply(lambda x: unidecode(x)).str.upper()
    df = df.set_index('Region')
    df = df[2019]
    return df


def loadDeathsAndRecoveries(regions, fromOS=False, inputDir='../data'):

    if fromOS is True:
        df = pl.load(join(inputDir, 'deathsAndRecoveries'), format='csv')
        regionsToDel = [r for r in regions if r not in df['Region'].unique()]
        coveredRegions = [r for r in regions if r not in regionsToDel]
        if len(regionsToDel) > 0:
            logging.warning(f'National Df doesn\'t include {len(regionsToDel)} regions (ignoring them):')
            logging.warning(f'{regionsToDel}')

        df = (df.set_index(['Region', 'Day'], drop=True).sort_index())
        df = df.drop_duplicates()
        return df, coveredRegions

    # Load db authentication data
    auth = loadConfig()

    # Set up DB client
    engine = create_engine(
        f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
    )
    logging.info('Connection established!')

    # Get dataset
    query = f'SELECT * FROM deathsRecoveriesActive;'
    df = pd.read_sql(query, con=engine)

    # Remove considered regions that are not in the dataset
    regionsToDel = [r for r in regions if r not in df['Region'].unique()]
    coveredRegions = [r for r in regions if r not in regionsToDel]
    if len(regionsToDel) > 0:
        logging.warning(f'National Df doesn\'t include {len(regionsToDel)} regions (ignoring them):')
        logging.warning(f'{regionsToDel}')

    df = (df.set_index(['Region', 'Day'], drop=True).sort_index())
    df = df.drop_duplicates()

    engine.dispose()

    return df, coveredRegions


def loadPredictions():
    # Load db authentication data
    auth = loadConfig()

    # Set up DB client
    engine = create_engine(
        f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
    )
    logging.info('Connection established!')

    # Find latest date
    query = 'SELECT inputDay FROM predictions ORDER BY Day DESC LIMIT 1;'
    latestinputDate = pd.read_sql(query, con=engine)
    latestinputDate = latestinputDate.loc[0]['inputDay']

    # Get dataset
    query = f"SELECT Day, Region, predictedCases FROM predictions WHERE inputDay=\'{latestinputDate}\';"
    df = pd.read_sql(query, con=engine)

    df = (df.set_index(['Region', 'Day'], drop=True).sort_index())
    engine.dispose()

    return df