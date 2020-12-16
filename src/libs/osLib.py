from os.path import join, exists, isfile, isdir, basename, normpath
from os import makedirs, listdir, rename, getcwd
from pathlib import Path

def getDir(path, name):
    dir = join(path, name)
    if exists(dir):
        return dir
    else:
        makedirs(dir)
        return dir

def fileExists(path, name):
    dir = join(path, name)
    if exists(dir):
        return True
    else:
        return False
