import cmsisdsp as dsp
import numpy as np


STATE_DIM = 19
MEAS_DIM = 8

x=np.zeros(STATE_DIM) #state vector
P = np.eye(STATE_DIM)*0.1 #covariance matrix with small initial uncertainty

def generate_sigma_points(x, P, alpha = 0.001, beta = 2, kappa = 0):
    n = len(x)
    lambda_ = alpha**2 * (n+kappa) - n
    sigma_points = np.zeros((2*n+1,n))


    sqrtP = np.zeros_like(P)
    status = dsp.arm_mat_cholesky_f32(P,sqrtP)

    sigma_points[0] = x
    for i in range(n):
        sigma_points[i+1] = x+sqrtP[i]
        sigma_points[n+i+1] = x-sqrtP[i]
    return sigma_points

def quaternion_rotate(q,v):
    q_conj = np.array([q[0],-q[1],-q[2],-q[3]])
    v_quat = np.array([0]+list(v))
    v_quat_prod = np.zeros_like(v_quat)
    dsp.arm_mat_cholesky_f32(q, v_quat, v_quat_prod, len(q))
    dsp.arm_mat_cholesky_f32()
