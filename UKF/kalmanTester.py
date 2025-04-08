import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pplot
import kalman_filter as ukf

#df = pd.read_csv("texasdata.csv")
#print(df.columns)


def fakeKalmanFilter(time, xa, ya, za, temp, pressure, rollRate, pitchRate, yawRate):
    return

def testFilter(inDataFile = "texasdatametric.csv"):

    state_csv = "state_vector.csv"
    inData = pd.read_csv(inDataFile)
    prev_time = 0
    dt_sum = 0
    times = inData.iloc[:,0]

    state = np.array([0, 0, 893,
                           0, 0, 0,
                           0.001, 0, 0,
                           1, 0, 0, 0,
                           0.001, 0, 0]) #small amount of roll, accel to avoid divide by zero in first quaternion process
    cov = np.eye(16)
    state_history = np.zeros((16, len(times)))
    cov_history = np.zeros((16, 16, len(times)))




    alpha = 1e-3
    beta = 2
    kappa = 1
    n = 16
    lambda_ = alpha**2 * (n + kappa) - n
    Wc = np.full(2*n + 1, 1. / (2*(n + lambda_)))
    Wm = np.full(2*n + 1, 1. / (2*(n + lambda_)))
    Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)

    process_noise = 1e-7*np.eye(n)
    process_noise[0:3, 0:3] += 1e-5 * np.eye(3)
    process_noise[6:9, 6:9] += 1e-3 * np.eye(3)
    process_noise[9:13, 9:13] += 1e-7 * np.eye(4)
    process_noise[13:16, 13:16] += 1e-6 * np.eye(3)

    baro_measurement_noise = np.diag(np.array((0.06, 0.01)))
    accel_measurement_noise = np.diag(9.8065*529000e-6*np.ones(3,))
    gyro_measurement_noise = np.diag(0.00225 * np.pi / 180 * np.ones(3, ))


    for ind, t in enumerate(times):
        row = inData.iloc[ind]
        dt = row.iloc[0]-prev_time
        dt_sum += dt
        #for col, value in row.items():
            #print(col, value)

        xa = (row.iloc[1] * math.cos(row.iloc[6])) + (row.iloc[2] * math.sin(row.iloc[6]))
        ya = 0
        za = (row.iloc[1] * math.sin(row.iloc[6])) + (row.iloc[2] * math.cos(row.iloc[6]))
        xa = row.iloc[2]
        za = row.iloc[1]
        

        #xa += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)
        #ya += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)
        #za += np.random.normal(0, 0.5 * 0.488 * 0.001 * 9.81)

        temp = row.iloc[7] #+ np.clip(np.random.normal(0, 0.25), -0.5, 0.5)
        pressure = row.iloc[8] #+ np.clip(np.random.normal(0, 75), -150, 150)
        

        rollRate = row.iloc[3] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))
        pitchRate = row.iloc[4] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))
        yawRate = row.iloc[5] #+ np.random.normal(0, 0.5 * 70 * 0.001 * (np.pi/180))
        

        #print("Index: ", ind, end = "\r")
        #print(np.array((pressure, temp)))

        
        state, cov, sigmas_f = ukf.predict(state, cov, Wm, Wc, process_noise, dt)
        
        state, cov = ukf.baro_update(state, cov, np.array((pressure, temp)), sigmas_f, Wm, Wc, baro_measurement_noise)
        #print("Baro: ", state[2])
        
        state, cov = ukf.accel_update(state, cov, np.array((xa,ya,za)), sigmas_f, Wm, Wc, accel_measurement_noise)
        #print("Accel: ", state[2])
        
        state, cov = ukf.gyro_update(state, cov, np.array((rollRate, pitchRate, yawRate)), sigmas_f, Wm, Wc, gyro_measurement_noise)
        #print("Gyro: ", state[2])
        
        
        with open(state_csv, "a") as f:
            np.savetxt(f, [state], delimiter=",")





        

        state_history[:, ind] = state
        cov_history[:, :, ind] = cov




        


testFilter()
print("done")

