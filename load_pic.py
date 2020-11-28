import os
import pickle

import numpy as np
import cv2
from utils import *


def CreateAllData(trainPath, testPath, vocabPath):
    s2lDict, l2sDict = CreateMap(vocabPath)
    trainData = CreatePicData(trainPath, s2lDict)
    testData = CreatePicData(testPath, s2lDict)
    saveTrain = 'train.dp'
    saveTest = 'test.dp'
    with open(saveTrain, 'wb') as f:
        pickle.dump(trainData, f)
    with open(saveTest, 'wb') as f:
        pickle.dump(testData, f)

def CreatePicData(dataPath, s2lDict):
    lines = open(dataPath, 'r').read().split('\n')
    lines = list(filter(lambda x:x!='', lines))

    ultra = []
    ultraLen = []
    label = []
    labelLen = []

    for path in lines:
        name = path.split(os.sep)[-1]
        name = name.split('_')[0]
        word = s2lDict[name]
        img = cv2.imread(path)
        height = img.shape[0]
        width = img.shape[1]
        print(img.shape)
        img = cv2.resize(img, (width // 2,  height// 2))
        print(img.shape)
        ultra.append(img)
        ultraLen.append(0)
        label.append(word)
        labelLen.append(0)

    ultra = np.array(ultra)
    ultra = ultra / 255.0
    ultraLen = np.array(ultraLen)
    label = np.array(label)
    labelLen = np.array(labelLen)

    return ultra, label, ultraLen, labelLen



