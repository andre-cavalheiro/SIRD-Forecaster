import os
import sys
sys.path.append(os.getcwd() + '/..')
import logging
import traceback
from os.path import join

from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Date, Float, Integer, String

import libs.pandasLib as pl
from libs.yamlLib import loadConfig


def updateCasesPerRegion(df, outputDir='../data'):
    dataTypes = {'Day': Date, 'Region': String(255), 'new-cases': Integer, 'total-cases': Integer}
    tableName = 'cases'
    df = df.reset_index()

    # Load db authentication data
    auth = loadConfig()

    try:
        # Set up DB client
        engine = create_engine(
            f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
        )
        logging.info('Connection established!')

        df.to_sql(tableName, con=engine, if_exists='append', index=False, dtype=dataTypes)
        pl.save(df, 'csv', join(outputDir, f'casesPerRegion'))
    except Exception as ex:
        logging.error(traceback.format_exc())

    engine.dispose()



def updateRecoveredAndDead(df, outputDir='../data'):

    dataTypes = {'Day': Date, 'Region': String(255), 'new-death': Float, 'new-recovered': Float,
                 'new-active-cases': Float, 'total-active-cases': Float}
    tableName = 'deathsRecoveriesActive'

    df = df.reset_index()
    df = df[['Day', 'Region', 'new-deaths', 'new-recovered', 'total-recovered', 'total-deaths',
             'new-active-cases', 'total-active-cases']]

    # Load db authentication data
    auth = loadConfig()

    try:
        # Set up DB client
        engine = create_engine(
            f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
        )
        logging.info('Connection established!')

        df.to_sql(tableName, con=engine, if_exists='append', index=True, dtype=dataTypes)
        pl.save(df, 'csv', join(outputDir, f'deathsAndRecoveries'))

    except Exception as ex:
        logging.error(traceback.format_exc())

    engine.dispose()


def updateForecast(df, outputDir='../data'):
    dataTypes = {'Day': Date, 'Region': String(255), 'predictedCases': Float, 'inputDay': Date, 'originalRegion': String(255)}
    tableName = 'predictions'

    df = df.reset_index()
    df = df[['Day', 'Region', 'predictedCases', 'originalRegion', 'inputDay']]

    # Load db authentication data
    auth = loadConfig()

    try:
        # Set up DB client
        engine = create_engine(
            f"mysql+pymysql://{auth['username']}:{auth['passwd']}@localhost:{auth['port']}/{auth['refinedDB']}"
        )
        logging.info('Connection established!')

        df.to_sql(tableName, con=engine, if_exists='append', index=False, dtype=dataTypes)
        pl.save(df, 'csv', join(outputDir, f'forecasts'))

    except Exception as ex:
        logging.error(traceback.format_exc())

    engine.dispose()
