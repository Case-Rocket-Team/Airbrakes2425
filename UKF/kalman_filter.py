import numpy as np
from scipy import linalg as sp
import pyquaternion as quaternion
alpha = 1e-3
beta = 2
kappa = 1

n = 16

lambda_ = alpha**2 * (n + kappa) - n




def sigma_points_fn(X, P):
    sigmas = np.zeros((2*n+1, n))
    U = sp.cholesky((n+lambda_)*P)

    sigmas[0] = X
    for k in range (n):
        sigmas[k+1] = X + U[k]
        sigmas[n+k+1] = X - U[k]
    return sigmas

def unscented_transform(sigmas, Wm, Wc, Q):
    x = np.dot(Wm, sigmas)
    kmax, n = sigmas.shape
    P = np.zeros((n,n))
    for k in range(kmax):
        y = sigmas[k]-x
        P+=Wc[k]*np.outer(y,y)
    P+=Q
    return x, P

def point_prop(state, dt):
    if(dt==0):
        dt = 0.001

    h_dt = dt**2 / 2

    pos = state[0:3]
    vel = state[3:6]
    accel = state[6:9]

    q = quaternion.Quaternion(state[9:13])
    q = q.normalised
    
    omega = state[13:16]

    
    pos_prime = pos + dt * vel + h_dt * accel
    vel_prime = vel + dt * accel
    accel_prime = accel

    rotated_q = quaternion.Quaternion(axis=dt*omega)*q
    rotated_q = rotated_q.normalised

    q_prime = rotated_q.elements
    
    omega_prime = omega
    
    return np.concatenate((pos_prime, vel_prime, accel_prime, q_prime, omega_prime))

def predict(X, P, Wm, Wc, Q, dt):
    sigmas = sigma_points_fn(X, P)
    num_sigmas = sigmas.shape[0]
    sigmas_f = np.zeros_like(sigmas)
    for i in range(num_sigmas):
        sigmas_f[i] = point_prop(sigmas[i], dt)
    
    xp, Pp = unscented_transform(sigmas_f, Wm, Wc, Q)

    return xp, Pp, sigmas_f

def baro_obs_pred(state):
    height = state[2]
    g = 9.80665  # Acceleration due to gravity (m/s^2)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)
    R = 8.31432  # Universal gas constant (J/(mol*K))
    T0 = 308.964  # Standard temperature at sea level (K)
    P0 = 101325  # Standard pressure at sea level (Pa)
    temperature = T0-0.0065*height
    base = 1 - (0.0065 * height) / T0
    if base <= 0 or not np.isfinite(base):
        print(f"[WARN] Invalid base in pressure calc: height={height}, T0={T0}, base={base}")
    pressure = P0 * np.power(base, (g * M) / (R * 0.0065))
    return np.array((pressure, temperature))

def baro_update(x, P, z, sigmas_f, Wm, Wc, R):
    sigmas_h = np.zeros((2*n+1, 2))
    num_sigmas = sigmas_h.shape[0]
    for i in range(num_sigmas):
        sigmas_h[i] = baro_obs_pred(sigmas_f[i])
    zp, Pz = unscented_transform(sigmas_h, Wm, Wc, R)
    #print(np.size(zp))
    #print(unscented_transform(sigmas_h, Wm, Wc, R))
    Pxz = np.zeros((np.size(x), np.size(z)))
    for i in range(num_sigmas):
        Pxz +=Wc[i]*np.outer(sigmas_f[i]-x, sigmas_h[i]-zp)
    
    K = np.dot(Pxz, np.linalg.inv(Pz))
    
    x_prime = x+np.dot(K, z-zp)
    P_prime = P+np.dot(np.dot(K,Pz), K.T)
    return x_prime, P_prime

def accel_obs_pred(state):
    accel = state[6:9]
    q = quaternion.Quaternion(state[9:13])
    q = q.normalised

    body_accel_q = q.conjugate*quaternion.Quaternion(axis=accel)*q
    body_accel = body_accel_q.vector

    return body_accel

def accel_update(x, P, z, sigmas_f, Wm, Wc, R):
    sigmas_h = np.zeros((2*n+1, 3))
    num_sigmas = sigmas_h.shape[0]
    for i in range(num_sigmas):
        sigmas_h[i] = accel_obs_pred(sigmas_f[i])
    zp, Pz = unscented_transform(sigmas_h, Wm, Wc, R)
    Pxz = np.zeros((np.size(x), np.size(z)))
    for i in range(num_sigmas):
        Pxz +=Wc[i]*np.outer(sigmas_f[i]-x, sigmas_h[i]-zp)
    
    K = np.dot(Pxz, np.linalg.inv(Pz))
    
    x_prime = x+np.dot(K, z-zp)
    P_prime = P+np.dot(np.dot(K,Pz), K.T)
    return x_prime, P_prime

def gyro_obs_pred(state):
    q = quaternion.Quaternion(state[9:13])
    q = q.normalised
    omega = state[13:16]

    body_omega_q = q.conjugate*quaternion.Quaternion(axis=omega)*q

    body_omega = body_omega_q.vector
    return omega

def gyro_update(x, P, z, sigmas_f, Wm, Wc, R):
    
    sigmas_h = np.zeros((2*n+1, 3))
    num_sigmas = sigmas_h.shape[0]
    for i in range(num_sigmas):
        sigmas_h[i] = gyro_obs_pred(sigmas_f[i])
    zp, Pz = unscented_transform(sigmas_h, Wm, Wc, R)
    
    Pxz = np.zeros((np.size(x), np.size(z)))
    for i in range(num_sigmas):
        Pxz +=Wc[i]*np.outer(sigmas_f[i]-x, sigmas_h[i]-zp)
    
    K = np.dot(Pxz, np.linalg.inv(Pz))
    

    x_prime = x+np.dot(K, z-zp)
    P_prime = P+np.dot(np.dot(K,Pz), K.T)
    return x_prime, P_prime
