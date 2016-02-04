import json
import os

def readSettingsFile(path):
    with open(path) as data_file:    
        data = json.load(data_file)
    return data

def processDataPathParameter(param):
    files = []
    if os.path.isdir(param):
        for f in os.listdir(param):
            if ".fit" in f:
                files.append(param.rstrip('/') + "/" + f)
    elif os.path.isfile(param):
        if ".fit" in param:
            files.append(param)
    return files
