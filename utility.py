import json
import os

def readSettingsFile(path):
    ''' 
    open a settings file and return a dict
    '''
    with open(path) as data_file:    
        data = json.load(data_file)
    return data

def processDataPathParameter(param):
    '''
    append valid files to list and return
    '''
    files = []
    if os.path.isdir(param):
        for f in os.listdir(param):
            if ".fit" in f or ".fits" in f:
                files.append(param.rstrip('/') + "/" + f)
    elif os.path.isfile(param):
        if ".fit" in param or ".fits" in param:
            files.append(param)
    return files
