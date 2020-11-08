from wav_func import *

corpusPath='F:\\dataset\\lip_move\\bigspan_sent'
mapPath='map_table.txt'
vocabPath='vocab.txt'
ultraMaxLen=200
sonicFreq=20000
lipOffset=40
nfft=8192

data = DealAllData(corpusPath, mapPath, vocabPath, ultraMaxLen, sonicFreq, lipOffset, nfft)
ToDestData(data)