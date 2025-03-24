import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pplot
import BetterUKF as ukf

#df = pd.read_csv("texasdata.csv")
#print(df.columns)


def fakeKalmanFilter(time, xa, ya, za, temp, pressure, rollRate, pitchRate, yawRate):
    return

def testFilter(inDataFile = "texasdatametric.csv"):
    inData = pd.read_csv(inDataFile)
    prev_time = 0
    dt_sum = 0
    times = inData.iloc[:,0]

    state = np.array([0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           1, 0, 0, 0,
                           0, 0, 0])

    cov = np.eye(16)
    state_history = np.zeros((16, len(times)))
    cov_history = np.zeros((16, 16, len(times)))


    

    for ind, t in enumerate(times):
        row = inData.iloc[ind]
        dt = row.iloc[0]-prev_time
        dt_sum += dt
        #for col, value in row.items():
            #print(col, value)

        xa = (row.iloc[4] * math.cos(row.iloc[5])) + (row.iloc[3] * math.sin(row.iloc[5]))
        ya = 0
        za = (row.iloc[4] * math.sin(row.iloc[5])) + (row.iloc[3] * math.cos(row.iloc[5]))

        #xa += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)
        #ya += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)
        #za += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)

        temp = row.iloc[9] #+ np.clip(np.random.normal(0, 0.25), -0.5, 0.5)
        pressure = row.iloc[10] #+ np.clip(np.random.normal(0, 75), -150, 150)

        rollRate = row.iloc[6] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))
        pitchRate = row.iloc[7] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))
        yawRate = row.iloc[8] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))

        print("Index: ", ind, end = "\r")
        state, cov = ukf.predict(state, cov, dt)
        state, cov = ukf.baro_correct(state, cov, np.vstack((pressure, temp)))
        state, cov = ukf.accel_correct(state, cov, np.vstack((xa,ya,za)))
        state, cov = ukf.gyro_correct(state, cov, np.vstack((rollRate, pitchRate, yawRate)))

        state_history[:, ind] = state
        cov_history[:, :, ind] = cov




        


testFilter()
print("done")

