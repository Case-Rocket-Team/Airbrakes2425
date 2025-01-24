import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import math
import matplotlib.pyplot as plt

def simulate_rocket_flight(ukf, steps, delta_t, thrust_function, mass_function):
    altitude = 6821.44  # Initial altitude in meters
    velocity = 279.93  # Initial velocity in m/s
    acceleration = 0  # Initialize acceleration
    apogee = 0 # Initialize apogee measurement
    apo_time = 0 # Initialize apo time
    time_list = [] # List of times for graph
    altitude_list = [] # List of altitudes for graph
    velocity_list = [] # List of velocities for graph
    acceleration_list = [] # List of accelerations for graph
    terminal_drag = 0 # Drag at terminal velocity



    for step in range(steps):
        current_time = step * delta_t

        time_list.append(current_time)
        altitude_list.append(altitude)
        velocity_list.append(velocity)
        acceleration_list.append(acceleration)

        # Get interpolated thrust and mass
        #thrust = thrust_function(current_time)  # Thrust in Newtons
        #mass = mass_function(current_time) * 0.0283495      # Mass in kg
        thrust = 0
        mass = 922.097 * 0.0283495

        if(math.isnan(thrust)):
            thrust = 0
            
        # Calculate thrust acceleration
        thrust_acceleration = thrust / mass  # Acceleration in m/s^2

        # Simulate realistic pressure and temperature based on altitude
        pressure = 101325 * np.exp(-altitude / 8400)  # Pressure in Pascals
        temperature = 15 - 0.0065 * altitude  # Approximate temperature in Celsius

        # Measurement: [altitude, acceleration]
        measurement = np.array([altitude, acceleration])

        # Run the UKF predict and update cycle
        ukf.predict(delta_t, pressure, temperature)
        ukf.update(measurement)

        # Print current state estimates
        print(f"Step {step+1}: Time = {current_time:.2f} s "
              f"Altitude = {ukf.state[0]:.2f} m, "
              f"Velocity = {ukf.state[1]:.2f} m/s, "
              f"Acceleration = {ukf.state[2]:.2f} m/s^2, "
              f"Thrust = {thrust:.2f} N")

        # Update actual altitude and velocity for simulation purposes
        if ukf.state[0]>apogee:
            apogee = ukf.state[0]
            apo_time = current_time
        density = ukf.calculate_density(pressure, temperature)
        drag = 0.5 * ukf.cdArea * density * velocity**2 / ukf.mass
        if velocity>0:
            net_acceleration = thrust_acceleration - 9.81 - drag  # Thrust - gravity - drag

        else:
            if abs(drag - 9.81)<0.001:
                terminal_drag = drag
            elif drag >9.81:
                drag = terminal_drag
            net_acceleration = thrust_acceleration - 9.81 + drag
            

        print(f"Net acceleration = {net_acceleration:.2f} m/s^2, "
              f"Drag = {drag:.2f} m/s^2")
        # Update position, velocity, and acceleration based on dynamics
        altitude += velocity * delta_t + 0.5 * net_acceleration * delta_t**2
        velocity += net_acceleration * delta_t
        acceleration = net_acceleration
        print(f"Altitude = {altitude:.2f} m")
        if altitude <-1:
            print("Rocket has hit the ground. Simulation ending.")
            print(f"Apogee: {apogee:.2f} m at "
                  f"time: {apo_time:.2f}")
            break
    plt.figure(figsize=(10, 6))
    plt.plot(time_list, altitude_list, label="Altitude", color="blue")
    plt.plot(time_list, velocity_list, label="Velocity", color="red")
    plt.plot(time_list, acceleration_list, label="Acceleration", color="yellow")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m), Velocity (m/s), Acceleration (m/s^2)")
    plt.title("Rocket Altitude vs. Time")
    plt.legend()
    plt.grid(True)
    plt.show()

class UKF:
    def __init__(self, initial_state, process_noise, measurement_noise, mass, cdArea):
        # State vector: [position, velocity, acceleration]
        self.state = initial_state  # np.array([position, velocity, acceleration])
        self.covariance = np.eye(3)  # Initial state covariance
        self.process_noise = process_noise  # Process noise matrix Q
        self.measurement_noise = measurement_noise  # Measurement noise matrix R
        self.mass = mass  # Mass of the rocket
        self.cdArea = cdArea # Drag coefficient multiplied by cross-sectional area
        self.gas_constant = 287.05  # Specific gas constant for dry air in J/(kg·K)

    def generate_sigma_points(self):
        n = self.state.size
        sigma_points = [self.state]
        sqrt_covariance = np.linalg.cholesky((n + 0.5) * self.covariance)

        for i in range(n):
            sigma_points.append(self.state + sqrt_covariance[:, i])
            sigma_points.append(self.state - sqrt_covariance[:, i])

        return np.array(sigma_points)

    def calculate_density(self, pressure, temperature):
        # Convert temperature to Kelvin if provided in Celsius
        temperature_kelvin = temperature + 273.15
        return pressure / (self.gas_constant * temperature_kelvin)

    def process_model(self, state, delta_t, density):
        position, velocity, acceleration = state

        # Calculate drag acceleration
        drag_acceleration = 0.5 * self.cdArea * density / self.mass * velocity**2

        # Net acceleration with drag
        net_acceleration = acceleration - drag_acceleration

        # Kinematic equations
        new_position = position + velocity * delta_t + 0.5 * net_acceleration * delta_t**2
        new_velocity = velocity + net_acceleration * delta_t

        return np.array([new_position, new_velocity, net_acceleration])

    def predict(self, delta_t, pressure, temperature):
        # Calculate air density
        density = self.calculate_density(pressure, temperature)

        # Generate sigma points
        sigma_points = self.generate_sigma_points()

        # Apply process model to each sigma point
        predicted_sigma_points = np.array([self.process_model(sp, delta_t, density) for sp in sigma_points])

        # Compute predicted mean
        self.state = np.mean(predicted_sigma_points, axis=0)

        # Compute predicted covariance
        self.covariance = self.process_noise.copy()
        for sp in predicted_sigma_points:
            diff = (sp - self.state).reshape(-1, 1)
            self.covariance += diff @ diff.T / len(predicted_sigma_points)

    def measurement_model(self, state):
        # Measurement: [altitude (position), acceleration]
        return np.array([state[0], state[2]])

    def update(self, measurement):
        # Generate sigma points and project to measurement space
        sigma_points = self.generate_sigma_points()
        measurement_sigma_points = np.array([self.measurement_model(sp) for sp in sigma_points])

        # Predicted measurement mean
        z_pred = np.mean(measurement_sigma_points, axis=0)

        # Measurement covariance S
        s_cov = self.measurement_noise.copy()
        for z in measurement_sigma_points:
            diff = (z - z_pred).reshape(-1, 1)
            s_cov += diff @ diff.T / len(measurement_sigma_points)

        # Cross covariance P_xz
        p_xz = np.zeros((3, 2))
        for sp, z in zip(sigma_points, measurement_sigma_points):
            state_diff = (sp - self.state).reshape(-1, 1)
            meas_diff = (z - z_pred).reshape(-1, 1)
            p_xz += state_diff @ meas_diff.T / len(sigma_points)

        # Calculate Kalman Gain
        k_gain = p_xz @ np.linalg.inv(s_cov)

        # Update state and covariance with measurement
        measurement_diff = (measurement - z_pred).reshape(-1, 1)
        self.state += (k_gain @ measurement_diff).flatten()
        self.covariance -= k_gain @ s_cov @ k_gain.T


thrust_data = pd.read_csv("src/ThrustCurve.csv")
mass_data = pd.read_csv("src/MassValues.csv")
cd_data = pd.read_csv("src/CdVals.csv")

thrust_function = CubicSpline(thrust_data["Time"],thrust_data["Thrust"],extrapolate=False)
mass_function = CubicSpline(mass_data["Time"],mass_data["Mass"])
cd_function = CubicSpline(cd_data["Time"],cd_data["Cd"], extrapolate = False)


initial_state = np.array([0.0, 0.0, 0.0])  # Starting at rest and ground level
process_noise = np.eye(3) * 0.1  # Process noise covariance
measurement_noise = np.eye(2) * 0.1  # Measurement noise covariance

# Rocket parameters
cd = 0.35  # Drag coefficient
area = 0.0173407967034  # Cross-sectional area in m^2
mass = 26.1  # Mass in kg

cdArea = 0.36*0.0173407967034

#cdArea = float(input("Enter C_d*Area: "))

# Instantiate the UKF
ukf = UKF(initial_state, process_noise, measurement_noise, mass, cdArea)

# Simulate flight with variable thrust and mass using the cubic spline functions
simulate_rocket_flight(ukf, steps=10000000, delta_t=0.01, thrust_function=thrust_function, mass_function=mass_function)
