import json


def load(fileName):
    with open(fileName, encoding="utf-8") as f:
        data = json.load(f)
    return data


def getSize(data):
    return len(data)


def dropByPercentage(data, percentage):
    assert(percentage>=0)
    assert(percentage<=1)
    stopIndex = int((1-percentage)*getSize(data))

    droppedData = data[stopIndex:]
    data = data[:stopIndex]
    return data, droppedData


def loop(data):
    iterator = -1
    while iterator+1 < getSize(data):
        iterator += 1
        currentItem = data[iterator]
        yield iterator, currentItem
