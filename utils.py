

def GetMapTable(mapPath):
    lines = open(mapPath, 'r').read().split('\n')
    lines = list(filter(lambda x:x!='', lines))
    mapDict = dict()
    for sinLine in lines:
        temp = sinLine.split(' ', 1)
        mapDict[temp[0]] = temp[1]
    return mapDict

def CreateMap(charFile):
    file = open(charFile)
    charTable = []
    charTable += file.read().split('\n')
    # charTable = [c.lower() for c in charTable]
    s2lDict = dict([(c, i) for i, c in enumerate(charTable)])
    l2sDict = dict([(i, c) for i, c in enumerate(charTable)])
    return  s2lDict, l2sDict