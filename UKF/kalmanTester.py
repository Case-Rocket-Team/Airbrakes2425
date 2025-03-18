import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pplot

#df = pd.read_csv("texasdata.csv")
#print(df.columns)

def fakeKalmanFilter(time, xa, ya, za, temp, pressure, rollRate, pitchRate, yawRate):
    return

def testFilter(inDataFile = "texasdata.csv"):
    inData = pd.read_csv(inDataFile)

    for i in range(1, len(inData)):
        row = inData.iloc[i]
        #for col, value in row.items():
            #print(col, value)

        xa = (row.iloc[4] * math.cos(row.iloc[5])) + (row.iloc[3] * math.sin(row.iloc[5]))
        ya = 0
        za = (row.iloc[4] * math.sin(row.iloc[5])) + (row.iloc[3] * math.cos(row.iloc[5]))

        fakeKalmanFilter(row.iloc[0], xa, ya, za, row.iloc[9], row.iloc[10], row.iloc[6], row.iloc[7], row.iloc[8])

testFilter()