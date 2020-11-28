from wav_func import *
from load_pic import *

corpusPath='F:\\dataset\\lip_move\\two_sent'
mapPath='map_table.txt'
vocabPath='vocab.txt'
ultraMaxLen=200
sonicFreq=20000
lipOffset=40
nfft=32768
saveDir = 'F:\\dataset\\ultra_pic_para'

trainPath = 'train.txt'
testPath = 'test.txt'

if __name__ == '__main__':
    data = DealAllData(corpusPath, mapPath, vocabPath, ultraMaxLen, sonicFreq, lipOffset, nfft, saveDir)
    # ToDestData(data)
    # CreateAllData(trainPath, testPath, vocabPath)
