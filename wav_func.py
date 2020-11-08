import glob
import os
import pickle

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt

from utils import *


def CutSTFTBiside(Zxx):
    # leftOff = 32
    leftOff =40
    rightOff = 12
    timeLen = Zxx.shape[1]
    Zxx = Zxx[:,leftOff:timeLen-rightOff]
    return Zxx

def FiltWav(signalWav, N, lowPass, highPass):
    b, a = signal.butter(N=N, Wn=[lowPass, highPass], btype='bandpass')
    filtSignal = signal.filtfilt(b, a, signalWav)  # signalWav为要过滤的信号
    return filtSignal

def LipGraphPara(sonicFreq, lipOffset, fs, nfft, padNum=5):
    span = fs / nfft
    basicPos = int(sonicFreq / fs * nfft)
    freqOffset = int(lipOffset // span) + padNum
    down = basicPos - freqOffset
    up = basicPos + freqOffset
    return down, up

def SingleSTFT(filePath, sonicFreq, lipOffset, nfft, frameTime=0.15, aheadTime=0.01):
    fs, signalWav = wav.read(filePath)

    lowPass = 2 * (sonicFreq - lipOffset) / fs
    highPass = 2 * (sonicFreq + lipOffset) / fs

    filtSignal = FiltWav(signalWav=signalWav, N=4, lowPass=lowPass, highPass=highPass)

    filtSignal = filtSignal / max(abs(filtSignal))

    signalNum = len(filtSignal)

    lmsSignal = filtSignal
    nperseg = int(fs * frameTime)
    noverlap = int(fs * (frameTime-aheadTime))
    f, t, Zxx = signal.stft(lmsSignal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Zxx = Zxx / nfft * 2
    Zxx = CutSTFTBiside(Zxx)
    # Zxx = CalPsd(Zxx)
    Zxx = np.abs(Zxx)

    t = t[0:Zxx.shape[1]]
    realOffset = lipOffset + 10
    down, up = LipGraphPara(sonicFreq, realOffset, fs, nfft)
    f = f[down:up]
    Zxx = Zxx[down:up,:]

    return t, f, Zxx

def GaussWeight(x, sigma):
    w = 1 / ((2 * np.pi) ** 0.5) * np.exp(-x*x/2/sigma/sigma)
    return w

def GaussSmooth(var, lenNum, sigma):
    varNum = len(var)
    varBefore = var[lenNum-1::-1]
    dataAfter = var[sigma-lenNum:]
    varAfter = dataAfter[::-1]
    var = np.concatenate((varBefore, var, varAfter))
    mid = lenNum//2
    coreIndex = [i - mid for i in range(lenNum)]
    coreIndex = np.array(coreIndex)
    w = GaussWeight(coreIndex, sigma)
    result = []
    # for i in range(mid, varNum-mid):
    #     temp = w*var[i-mid, i+]

    for i in range(lenNum, varNum+lenNum):
        temp = np.dot(w,var[i-mid:i+lenNum-mid])
        result.append(temp)
    return np.array(result)

def CheckGradActive(Zxx):
    time = Zxx.shape[1]

    # sumRes = np.var(Zxx, axis=0)
    Zxx = Zxx ** 2
    sumRes = np.sum(Zxx, axis=0)

    lenNum = 10
    sigma = 4
    sumRes = GaussSmooth(sumRes, lenNum, sigma)

    topValue = 2000
    maxValue = np.max(sumRes)
    rate = maxValue / topValue
    sumRes = sumRes / rate

    peakIdx = signal.find_peaks(sumRes)
    peaks = sumRes[peakIdx[0]]
    peaks = sorted(peaks, reverse=True)
    peakAvg = np.mean(peaks[0:5])
    # print('ten divide of peakavg', peakAvg)

    activation = np.zeros((time,))

    winLen = 8
    maxDiff = 60
    # lowThresh = np.mean(sumRes) * 0.95
    lowThresh = peakAvg / 10

    for i in range(0, time - winLen):
        curWin = sumRes[i:i+winLen]
        curMean = np.mean(curWin)
        curMax = np.max(curWin)
        if curMax - curMean > maxDiff or curMean > lowThresh:
            activation[i:i+winLen] = 1
        # if curMean > lowThresh:
        #     activation[i:i+winLen] = 1

    plt.plot(sumRes)
    plt.plot(activation*1000)
    plt.show()

    return activation

def SplitIndex(activation, winLen):
    time = activation.shape[0]
    upBound = 2
    deCount = upBound
    start = -1
    index = []
    for i in range(time-winLen):
        window = activation[i:i+winLen]
        judge = (window == 0)
        # print(judge)
        if False not in judge:
            deCount += 1
            if deCount == 2:
                end = min(i + winLen - upBound, time-1)
                index.append((start, end))
                start = -1
            deCount = min(deCount, upBound)
        else:
            if deCount >= upBound:
                start = i
            deCount = 0
    if start != -1:
        index.append((start, time-1))
    return index

def SplitWord(Zxx, indexClips):
    words = []
    for clip in indexClips:
        words.append(Zxx[:, clip[0]:clip[-1]])
    return words

def PadUltra(words, ultraMaxLen):
    padData = []
    allLen = []
    for sinData in words:
        wordLen = sinData.shape[1]

        padLen = ultraMaxLen - wordLen
        pad = np.zeros((sinData.shape[0], padLen))
        sinData = np.concatenate((sinData, pad), axis=1)

        padData.append(sinData)
        allLen.append(wordLen)
    return padData

def Str2Label(words, s2lDict):
    labels = []
    for w in words:
        l = s2lDict[w]
        labels.append(l)
    return labels

def SubAdjacent(Zxx):
    time = Zxx.shape[1]
    shape = Zxx.shape
    result = np.zeros((Zxx.shape[0], time-1))
    for i in range(1, time):
        result[:, i - 1] = (Zxx[:, i] - Zxx[:, i - 1])
    return result

def CheckSplit(filePath, mapDict, sonicFreq, lipOffset, nfft):
    fileName = os.path.splitext(filePath.split(os.sep)[-1])[0]
    fileName = fileName.split('_')[0]
    fileName = fileName.lower()

    sent = mapDict[fileName]
    words = sent.split(' ')
    sentLen = len(words)
    print(sent)

    t, f, Zxx = SingleSTFT(filePath, sonicFreq, lipOffset, nfft)
    Zxx = SubAdjacent(Zxx)
    activation = CheckGradActive(Zxx)
    winLen = 10
    indexClips = SplitIndex(activation, winLen)
    print(indexClips)

    if len(indexClips) != sentLen:
        print(f'split not correct at {filePath}')
        return None
    else:

        splitWords = SplitWord(Zxx, indexClips)
        return splitWords, words

def DealAllData(corpusPath, mapPath, vocabPath, ultraMaxLen, sonicFreq, lipOffset, nfft):

    realUltra = []
    realLabel = []
    realUltraLen = []
    realLabelLen = []

    allCount = 0
    failCount = 0
    pattern = '*'
    mapDict = GetMapTable(mapPath)
    s2lDict, l2sDict = CreateMap(vocabPath)
    searchPath = os.path.join(corpusPath, pattern)

    savedPath = 'tempData.dp'

    if not os.path.exists(savedPath):
        # 分割错误的数据
        errorPath = 'error_split.txt'
        file = open(errorPath, 'w')

        for path in glob.glob(searchPath):
            data = CheckSplit(path, mapDict, sonicFreq, lipOffset, nfft)
            if data == None:
                failCount += 1
                file.write(path)
                file.write('\n')
            else:
                ultra = data[0]
                label = data[1]
                ultra = PadUltra(ultra, ultraMaxLen)
                label = Str2Label(label, s2lDict)
                realUltra += ultra
                realLabel += label

            allCount += 1
        print("fail split rate is", failCount / allCount)
        file.close()

        with open('tempData.dp', 'wb') as f:
            tempData = [realUltra, realLabel]
            pickle.dump(tempData, f)
    else:
        with open('tempData.dp', 'rb') as f:
            tempData = pickle.load(f)
            realUltra = tempData[0]
            realLabel = tempData[1]

    vocabLen = len(s2lDict)
    print(vocabLen)
    classData = []
    for i in range(vocabLen):
        classData.append([])
    for i in range(len(realLabel)):
        classIdx = realLabel[i]
        classData[classIdx].append(realUltra[i])
    trainPercent = 0.7
    trainUltra = []
    trainLabel = []
    testUltra = []
    testLabel = []

    for i in range(vocabLen):
        curLen = len(classData[i])
        trainNum = int(curLen * trainPercent)
        trainUltra += classData[i][0:trainNum]
        testUltra += classData[i][trainNum:]
        trainLabel += [i] * trainNum
        testLabel += [i] * (curLen - trainNum)
    return [trainUltra, trainLabel, testUltra, testLabel]


def ToDestData(data):
    trainUltra = data[0]
    trainLabel = data[1]
    testUltra = data[2]
    testLabel = data[3]
    trainUltraLen = np.zeros(len(trainUltra))
    trainLabelLen = np.zeros(len(trainLabel))
    testUltraLen = np.zeros(len(testUltra))
    testLabelLen = np.zeros(len(testLabel))
    # 转换成numpy
    trainUltra = np.array(trainUltra)
    trainLabel = np.array(trainLabel)
    testUltra = np.array(testUltra)
    testLabel = np.array(testLabel)
    trainUltraLen = np.array(trainUltraLen)
    trainLabelLen = np.array(trainLabelLen)
    testUltraLen = np.array(testUltraLen)
    testLabelLen = np.array(testLabelLen)
    trainData = [trainUltra, trainLabel, trainUltraLen, trainLabelLen]
    testData = [testUltra, testLabel, testUltraLen, testLabelLen]
    ##
    saveTrain = 'train.dp'
    saveTest = 'test.dp'
    with open(saveTrain, 'wb') as f:
        pickle.dump(trainData, f)
    with open(saveTest, 'wb') as f:
        pickle.dump(testData, f)










