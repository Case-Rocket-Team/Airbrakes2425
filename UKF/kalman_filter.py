import cmsisdsp as dsp
import numpy as np
import matplotlib.pyplot as plt
STATE_DIM = 16 #[x,y,z,v_x,v_y,v_z,a_x,a_y,a_z,q_w,q_x,q_y,q_z,w_x,w_y,w_z]
MEAS_DIM = 7 #[z,abody_x,abody_y,abody_z,w_x,w_y,w_z]
alpha = 1e-3
beta = 2
kappa = 1

process_noise = 1e-7*np.eye(STATE_DIM)
process_noise[0:3, 0:3] += 1e-5 * np.eye(3)
# process_noise[3:6, 3:6] += 1e-4 * np.eye(3)
process_noise[6:9, 6:9] += 1e-3 * np.eye(3)

process_noise[9:13, 9:13] += 1e-7 * np.eye(4)
process_noise[13:16, 13:16] += 1e-6 * np.eye(3)

starting_state = np.array([0,0,0,
                   0,0,0,
                   0,0,0,
                   1,0,0,0,
                   0,0,0])

baro_measurement_noise = np.diag(np.array((0.06, 0.01)))
accel_measurement_noise = np.diag(9.8065*529000e-6*np.ones(3,))
gyro_measurement_noise = np.diag(0.00225 * np.pi / 180 * np.ones(3, ))

baro_noise_lerp = 0
accel_noise_lerp = 0
gyro_noise_lerp = 0

def sigma_gen(state, covariance):
    num_states = np.size(state)
    sigma_points = np.zeros((num_states, 2 * num_states + 1))
    mean_weights = np.zeros(2 * num_states + 1)
    cov_weights = np.zeros(2 * num_states + 1)

    sigma_points[:,0] = state
    mean_weights[0] = (alpha**2 * kappa - num_states) / (alpha**2 * kappa)
    cov_weights[0] = mean_weights[0] - alpha**2 + beta

    A = dsp.arm_mat_cholesky_f32(covariance)[1]

    remaining_weights = 1 / (2 * alpha**2 * kappa)

    for i in range(num_states):
      sigma_points[:,1 + i] = state + alpha * dsp.arm_sqrt_f32(kappa)[1] * A[:,i]
      sigma_points[:,1 + num_states + i] = state - alpha * dsp.arm_sqrt_f32(kappa)[1] * A[:,i]

      mean_weights[1 + i:: num_states] = remaining_weights
      cov_weights[1 + i:: num_states] = remaining_weights

    return sigma_points, mean_weights, cov_weights


def point_prop(state, dt):
   h_dt = dt**2/2
   pos = state[0:3]

   vel = state[3:6]
   acc = state[6:9]
   q = dsp.arm_mat_trans_f32(state[9:13])[1] 
   #q = dsp.arm_quaternion_normalize_f32(q)

   omega = state[13:16]

   pos_prime = pos+dt*vel+h_dt*acc
   vel_prime = vel+dt*acc
   acc_prime = acc

   #BUG CMSIS transpose is not working, doing numpy for now
   q_prime = np.zeros_like(q)
   for i, omega_row in enumerate(omega):
      theta = (np.sqrt(omega_row[0]**2 + omega_row[1]**2+omega_row[2]**2))*dt
      R = np.eye(3) + np.array([[0, -1*omega_row[2], omega_row[1]], [omega_row[2], 0, -1*omega_row[0]], [-1*omega_row[1], omega_row[0], 0]])+ (1- np.cos(theta))*np.array([[omega_row[0]**2, omega_row[0]*omega_row[1], omega_row[0]*omega_row[2]], [omega_row[1]*omega_row[0], omega_row[1]**2, omega_row[1]*omega_row[2]], [omega_row[2]*omega_row[0], omega_row[2]*omega_row[1], omega_row[2]**2]])
      rotated_q = dsp.arm_quaternion_product_single_f32(dsp.arm_rotation2quaternion_f32(R),q[i])
      rotated_q = dsp.arm_quaternion_normalize_f32(rotated_q)
      q_prime[i] = rotated_q

   q_prime = q_prime.T


   omega_prime = omega
   return np.concatenate((pos_prime,vel_prime,acc_prime,q_prime,omega_prime))

def update_process_noise(innovation, kalman_gain, process_noise, noise_lerp):
   
   vec = dsp.arm_mat_mult_f32(kalman_gain, innovation)[1]
   vec_T = dsp.arm_mat_trans_f32(vec)[1]

   return (1-noise_lerp)*process_noise+noise_lerp*(dsp.arm_mat_mult_f32(vec_T, vec)[1])

def predict(state, covariance, dt):
   sigma_points, mean_weights, cov_weights = sigma_gen(state, covariance)
   x_prime = point_prop(sigma_points,dt)
   #mean = dsp.arm_mat_mult_f32(mean_weights, x_prime.T)[1]
   print(f"X_prime: {np.size(x_prime.T)}")
   print(f"Mean weights: {np.size(mean_weights)}")
   mean = mean_weights@x_prime.T
   dev = x_prime - mean[:, np.newaxis]
   new_cov = np.zeros((dev.shape[0], dev.shape[0]))
   for i in range(dev.shape[1]):
      d = dev[:,i]
      new_cov+=cov_weights[i]*np.outer(d,d)
   
   covariance+=process_noise

   return mean, covariance

def baro_obs_pred(state):
   height = state[2]
   g = 9.80665  # Acceleration due to gravity (m/s^2)
   M = 0.0289644  # Molar mass of Earth's air (kg/mol)
   R = 8.31432  # Universal gas constant (J/(mol*K))
   T0 = 288.15  # Standard temperature at sea level (K)
   P0 = 101325  # Standard pressure at sea level (Pa)

   temperature = T0-0.0065*height
   pressure = P0 *((1-(0.0065*height)/T0)**(g*M)/(R*0.0065))/1e3


   return np.vstack((pressure, temperature))

def update_baro_measurement_noise(residual, pred_measure_cov):
   return (1-baro_noise_lerp)*baro_measurement_noise + baro_noise_lerp*(dsp.arm_mat_mult_f32(residual, dsp.arm_mat_trans_f32(residual)[1])[1] + pred_measure_cov)

def baro_correct(state, covariance, measurement):
   global baro_measurement_noise
   global process_noise
   sigma_points, mean_weights, cov_weights = sigma_gen(state, covariance)
   pred_measurements = baro_obs_pred(sigma_points)

   mean_measure = dsp.arm_mat_mult_f32(mean_weights, dsp.arm_mat_trans_f32(pred_measurements)[1])[1]
   raw_cov_measure = 0
   cross_cov = 0
   for i in range(2*np.size(state)+1):
      dev = pred_measurements[i]-mean_measure
      raw_cov_measure +=cov_weights[i]*(dsp.arm_mat_mult(dev, dsp.arm_mat_trans_f32(dev)[1])[1])
      sigma_dev = sigma_points[i] - mean_measure
      cross_cov +=cov_weights[i]*(dsp.arm_mat_mult_f32(sigma_dev, dsp.arm_mat_trans_f32(dev)[1])[1])


   cov_measure = raw_cov_measure+baro_measurement_noise
   kalman_gain = dsp.arm_mat_mult_f32(cross_cov, dsp.arm_mat_inverse_f32(cov_measure)[1])[1]
   posterior_x = state+dsp.arm_mat_mult_f32(kalman_gain, measurement-mean_measure)[1]
   posterior_cov = covariance - dsp.arm_mat_mult_f32(kalman_gain, dsp.arm_mat_mult_f32(cov_measure, dsp.arm_mat_trans_f32(kalman_gain)[1])[1])[1]

   innovation = measurement-mean_measure
   residual = measurement-baro_obs_pred(dsp.arm_mat_trans_f32(posterior_x))[1]
   baro_measurement_noise = update_baro_measurement_noise(residual, raw_cov_measure)
   process_noise = update_process_noise(innovation, kalman_gain,process_noise,baro_noise_lerp)
   posterior_cov = (posterior_cov + dsp.arm_mat_trans_f32(posterior_cov)[1])/2

   return posterior_x, posterior_cov


def accel_obs_pred(state):
   g=9.8065

   accel = state[6:9]+dsp.arm_mat_trans_f32(np.array([[0,0,g]]))[1] #TODO check if Sirin IMU includes gravity in measurement
   q = dsp.arm_mat_trans_f32(state[9:13])[1]
   q = dsp.arm_quaternion_normalize_f32(q)


   #TODO check if these are correct, and if omega is needed
   #omega = dsp.arm_quaternion_product_single(dsp.arm_quaternion_conjugate_f32(q),dsp.arm_quaternion_product_single(dsp.arm_mat_trans_f32(state[13:16]),q))[1:] 
   body_acc = dsp.arm_quaternion_product_single(dsp.arm_quaternion_conjugate_f32(q),dsp.arm_quaternion_product_single(dsp.arm_mat_trans_f32(accel),q)[1])[1:]

   #should we remove centripetal accel? will that provide a ton of error? how far from c_g is Sirin's IMU?

   return body_acc

def update_accel_measurement_noise(residual, pred_measure_cov):
   return (1-accel_noise_lerp)*accel_measurement_noise + accel_noise_lerp*(dsp.arm_mat_mult_f32(residual, dsp.arm_mat_trans_f32(residual))[1] + pred_measure_cov)


def accel_correct(state, covariance, measurement):
   global accel_measurement_noise
   global process_noise
   sigma_points, mean_weights, cov_weights = sigma_gen(state, covariance)
   pred_measurements = accel_obs_pred(sigma_points)

   mean_measure = dsp.arm_mat_mult_f32(mean_weights, dsp.arm_mat_trans_f32(pred_measurements)[1])[1]
   raw_cov_measure = 0
   cross_cov = 0
   for i in range(2*np.size(state)+1):
      dev = pred_measurements[i]-mean_measure
      raw_cov_measure +=cov_weights[i]*(dsp.arm_mat_mult(dev, dsp.arm_mat_trans_f32(dev)[1])[1])
      sigma_dev = sigma_points[i] - mean_measure
      cross_cov +=cov_weights[i]*(dsp.arm_mat_mult_f32(sigma_dev, dsp.arm_mat_trans_f32(dev)[1])[1])


   cov_measure = raw_cov_measure+accel_measurement_noise
   kalman_gain = dsp.arm_mat_mult_f32(cross_cov, dsp.arm_mat_inverse_f32(cov_measure)[1])[1]
   posterior_x = state+dsp.arm_mat_mult_f32(kalman_gain, measurement-mean_measure)[1]
   posterior_cov = covariance - dsp.arm_mat_mult_f32(kalman_gain, dsp.arm_mat_mult_f32(cov_measure, dsp.arm_mat_trans_f32(kalman_gain)[1])[1])[1]

   innovation = measurement-mean_measure
   residual = measurement-accel_obs_pred(dsp.arm_mat_trans_f32(posterior_x)[1])
   accel_measurement_noise = update_accel_measurement_noise(residual, raw_cov_measure)
   process_noise = update_process_noise(innovation, kalman_gain, process_noise, accel_noise_lerp)
   posterior_cov = (posterior_cov + dsp.arm_mat_trans_f32(posterior_cov)[1])/2

   return posterior_x, posterior_cov

def gyro_obs_pred(state):
   q = dsp.arm_mat_trans_f32(state[9:13])[1]
   q = dsp.arm_quaternion_normalize_f32(q)
   body_omega = dsp.arm_quaternion_product_single(dsp.arm_quaternion_conjugate_f32(q),dsp.arm_quaternion_product_single(dsp.arm_mat_trans_f32(state[13:16])[1],q))[1:]
   return body_omega

def update_gyro_measurement_noise(residual, pred_measure_cov):
   return (1-gyro_noise_lerp)*gyro_measurement_noise + gyro_noise_lerp*(dsp.arm_mat_mult_f32(residual, dsp.arm_mat_trans_f32(residual)[1])[1] + pred_measure_cov)


def gyro_correct(state, covariance, measurement):
   global gyro_measurement_noise
   global process_noise
   sigma_points, mean_weights, cov_weights = sigma_gen(state, covariance)
   pred_measurements = gyro_obs_pred(sigma_points)

   mean_measure = dsp.arm_mat_mult_f32(mean_weights, dsp.arm_mat_trans_f32(pred_measurements)[1])[1]
   raw_cov_measure = 0
   cross_cov = 0
   for i in range(2*np.size(state)+1):
      dev = pred_measurements[i]-mean_measure
      raw_cov_measure +=cov_weights[i]*(dsp.arm_mat_mult(dev, dsp.arm_mat_trans_f32(dev)[1])[1])
      sigma_dev = sigma_points[i] - mean_measure
      cross_cov +=cov_weights[i]*(dsp.arm_mat_mult_f32(sigma_dev, dsp.arm_mat_trans_f32(dev)[1])[1])


   cov_measure = raw_cov_measure+gyro_measurement_noise
   kalman_gain = dsp.arm_mat_mult_f32(cross_cov, dsp.arm_mat_inverse_f32(cov_measure)[1])[1]
   posterior_x = state+dsp.arm_mat_mult_f32(kalman_gain, measurement-mean_measure)[1]
   posterior_cov = covariance - dsp.arm_mat_mult_f32(kalman_gain, dsp.arm_mat_mult_f32(cov_measure, dsp.arm_mat_trans_f32(kalman_gain)[1])[1])[1]

   innovation = measurement-mean_measure
   residual = measurement-gyro_obs_pred(dsp.arm_mat_trans_f32(posterior_x)[1])
   gyro_measurement_noise = update_gyro_measurement_noise(residual, raw_cov_measure)
   process_noise = update_process_noise(innovation, kalman_gain, process_noise, gyro_noise_lerp)
   posterior_cov = (posterior_cov + dsp.arm_mat_trans_f32(posterior_cov)[1])/2

   return posterior_x, posterior_cov


   
