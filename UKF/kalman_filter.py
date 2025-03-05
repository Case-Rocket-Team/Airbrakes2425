import cmsisdsp as dsp
import numpy as np


STATE_DIM = 16 #[x,y,z,v_x,v_y,v_z,a_x,a_y,a_z,q_w,q_x,q_y,q_z,w_x,w_y,w_z]
MEAS_DIM = 7 #[z,abody_x,abody_y,abody_z,w_x,w_y,w_z]

x=np.zeros(STATE_DIM) #state vector
P = np.eye(STATE_DIM)*0.1 #covariance matrix with small initial uncertainty

#alpha beta and kappa are ideal values, should be tweaked with testing
alpha = 0.001 #determines spread of sigma points around mean
beta = 2 #incorporates prior knowledge of x distribution
kappa = 0 #secondary scaling parameter
lambda_ = alpha**2*(STATE_DIM+kappa) - STATE_DIM #scaling parameter

#Covariance and mean weights
W_m = np.full(2 * STATE_DIM + 1, 1 / (2 * (STATE_DIM + lambda_)))
W_c = np.copy(W_m)
W_m[0] = lambda_/(STATE_DIM+lambda_)
W_c[0] = lambda_/(STATE_DIM+lambda_) + (1-alpha**2+beta)


#generate the sigma points 
def generate_sigma_points(x, P):
    n = STATE_DIM
    sigma_points = np.zeros((2*n+1,n))
    sqrtP = np.zeros_like(P)
    dsp.arm_mat_cholesky_f32(P,sqrtP)

    sigma_points[0] = x
    for i in range(n):
        sigma_points[i+1] = x+sqrtP[i]
        sigma_points[n+i+1] = x-sqrtP[i]
    return sigma_points


#calculate dq from q and omega
def quaternion_derivative(q, omega):
    q_w, q_x, q_y, q_z=q
    omega_x, omega_y, omega_z = omega

    dq = 0.5*np.array([-q_x * omega_x - q_y * omega_y - q_z * omega_z,
         q_w * omega_x + q_y * omega_z - q_z * omega_y,
         q_w * omega_y - q_x * omega_z + q_z * omega_x,
         q_w * omega_z + q_x * omega_y - q_y * omega_x])
    return dq

#predictor model (dynamics equations)
def predict_state(x, dt):
    pos = x[:3]
    vel = x[3:6]
    q = x[6:10]
    omega = x[10:13]
    acc = x[13:16]

    new_pos = pos + vel*dt+0.5*acc*(dt**2) #kinematic position eq
    new_vel = vel + acc*dt #kinematic velocity eq
    dq = quaternion_derivative(q, omega) #calculate dq
    new_q = q+dq*dt #calculate new q with dq and dt
    dsp.arm_quaternion_normalize_f32(new_q, new_q, len(new_q)) #normalize new quaternion vector

    new_omega = omega
    new_acc = acc

    new_state = np.zeros_like(x)
    new_state[:3] = new_pos
    new_state[3:6] = new_vel
    new_state[6:10] = new_q
    new_state[10:13] = new_omega
    new_state[13:16] = new_acc

    return new_state #predicted state vector for next time step

#transform prediction to measurement space
def measurement_model(x):
    q=x[6:10]
    acc_world = x[13:16]
    omega = x[10:13]

    #transform acceleration from world frame (state vector) to body frame (IMU)
    acc_body = np.zeros(3)
    inverse_q = np.zeros_like(q)
    dsp.arm_quaternion_inverse_f32(q,inverse_q,len(q))
    dsp.arm_quaternion_product(acc_world, inverse_q, acc_body)

    return np.concatenate((x[4], acc_body, omega))


#predictor
def ukf_predict(x, P, dt):

    #generate and propagate sigma points through prediction model
    sigma_points = generate_sigma_points(x, P)
    propagated_sigma_points = np.array([predict_state(sp,dt) for sp in sigma_points])

    #predict new state vector and covariance
    x_pred = np.sum(W_m[:,None] * propagated_sigma_points, axis=0)
    P_pred = np.zeros(STATE_DIM, STATE_DIM)
    for i in range (2*STATE_DIM+1):
        diff = propagated_sigma_points[i] -x_pred
        P_pred += W_c[i]*np.outer(diff,diff)
    return x_pred, P_pred


#corrector
def ukf_update(x_pred, P_pred, z):

    #generate new sigma points and propagate through measurement model
    sigma_points = generate_sigma_points(x_pred, P_pred)
    measurement_sigma = np.array([measurement_model(sp) for sp in sigma_points])

    #predict measurement
    z_pred = np.sum(W_m[:,None]*measurement_sigma, axis = 0)

    P_zz = np.zeros((MEAS_DIM, MEAS_DIM))
    P_xz = np.zeros((STATE_DIM, MEAS_DIM))

    for i in range (2*STATE_DIM+1):
        z_diff = measurement_sigma[i]-z_pred #difference between measurement and predicted measurement
        x_diff = sigma_points[i] - x_pred #difference between current state and predicted state
        P_zz += W_c[i]*np.outer(z_diff, z_diff) #covariance of z and z (will need to replace np.outer with CMSIS equivalent eventually)
        P_xz +=W_c[i]*np.outer(x_diff,z_diff) #covariance of x and x (will need to replace np.outer here too)

    #calculate Kalman gain
    K = np.zeros((STATE_DIM, MEAS_DIM))
    P_zz_inverse = np.zeros_like(P_zz)
    dsp.arm_mat_inverse_f32(P_zz, P_zz_inverse)
    dsp.arm_mat_mult_f32(P_xz, P_zz_inverse, K)

    #calculate corrected state vector
    x_updated = x_pred + K*(z-z_pred)

    #calculate corrected covariance
    k_pzz = np.zeros((STATE_DIM, MEAS_DIM))
    dsp.arm_mat_mult_f32(K, P_zz, k_pzz)
    K_t = np.zeros((MEAS_DIM, STATE_DIM))
    dsp.arm_mat_trans_f32(K, K_t)
    k_pzz_kt = np.zeros((STATE_DIM, STATE_DIM))
    dsp.arm_mat_mult_f32(k_pzz, K_t, k_pzz_kt)
    P_updated = P_pred - k_pzz_kt

    return x_updated, P_updated

#takes in sensor measurements and formats them for the Kalman filter (main component is calculating baro alt)
def get_measurement(acc, baro, gyro, P_0):
    baro_temp = baro[0]
    baro_pres = baro[1]
    h = (((P_0/baro_pres)**(1/5.257)-1)*(baro_temp+273.15))/(0.0065) #hypsometric formula, probably incorrect but can fix later
    return np.concatenate((h, acc, gyro)) #measurement vector


