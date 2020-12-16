
import yaml


def loadConfig(file='auth.yaml'):
    with open(file) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content
