#find vertical acceleration with gyro + accelerometer
#run kinematic simulation to find estimated apogee










import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import math
import matplotlib.pyplot as plt

def calculate_density(pressure, temperature):
    # Convert temperature to Kelvin if provided in Celsius
    temperature_kelvin = temperature + 273.15
    return pressure / (287.05 * temperature_kelvin)

def simulate_rocket_flight(steps, delta_t, startingAlt, startingDist, startingSpeed, startingAngleH, cdArea):
    altitude = startingAlt  # Initial altitude in meters
    distance = startingDist
    velocity = [startingSpeed * math.cos(startingAngleH), startingSpeed * math.sin(startingAngleH)] #velocity vector using initial gyro angle above horizontal
    speed = startingSpeed
    acceleration = [0, 0] # Initialize acceleration
    apogee = 0 # Initialize apogee measurement
    apo_time = 0 # Initialize apo time
    time_list = [] # List of times for graph
    altitude_list = [] # List of altitudes for graph
    distance_list = [] # List of distances for graph
    speed_list = [] # List of velocities for graph
    acceleration_list = [] # List of accelerations for graph
    terminal_drag = 0 # Drag at terminal velocity



    for step in range(steps):
        current_time = step * delta_t

        time_list.append(current_time)
        altitude_list.append(altitude)
        speed_list.append(speed)
        acceleration_list.append(acceleration)
        distance_list.append(distance)

        thrust = 0
        mass = 25.57

        if(math.isnan(thrust)):
            thrust = 0
            
        # Calculate thrust acceleration


        # Simulate realistic pressure and temperature based on altitude
        pressure = 101325 * np.exp(-altitude / 8400)  # Pressure in Pascals
        temperature = 15 - 0.0065 * altitude  # Approximate temperature in Celsius


        

        # Print current state estimates
        #print(f"Step {step+1}: Time = {current_time:.2f} s "
         #     f"Altitude = {altitude:.2f} m, "
          #    f"speed = {speed:.2f} m/s, "
           #   f"Acceleration = {acceleration:.2f} m/s^2, "
            #  f"Thrust = {thrust:.2f} N")

        # Update actual altitude and velocity for simulation purposes
        if altitude > apogee:
            apogee = altitude
            apo_time = current_time
        
        #update drage
        density = calculate_density(pressure, temperature)
        drag = 0.5 * cdArea * density * speed**2 / mass

        #update acceleration vector
        if speed > 0:
            acceleration[0] = drag * math.cos(startingAngleH)
            acceleration[1] = -9.81 - (drag * math.sin(startingAngleH))
        else:
            if abs(drag - 9.81) < 0.001:
                terminal_drag = drag
            elif drag > 9.81:
                drag = terminal_drag
            acceleration[0] = drag * math.cos(startingAngleH)
            acceleration[1] = -9.81 + (drag * math.sin(startingAngleH))

        #update velocity vector
        velocity[0] += acceleration[0] * delta_t
        velocity[1] += acceleration[1] * delta_t

        #recalculate new speed and acceleration magnitude
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        net_acceleration = math.sqrt(acceleration[0]**2 + acceleration[1]**2)

        # Update position, speed, and acceleration based on dynamics...WRONG its kinematics
        distance += velocity[0] * delta_t + 0.5 * acceleration[0] * delta_t**2
        altitude += velocity[1] * delta_t + 0.5 * acceleration[1] * delta_t**2
        speed = net_acceleration * delta_t
        #print(f"Altitude = {altitude:.2f} m")
        if altitude <-1:
            print("Rocket has hit the ground. Simulation ending.")
            print(f"Apogee: {apogee:.2f} m at "
                  f"time: {apo_time:.2f}")
            break
    
    data = {
        'Time (s)': time_list,
        'Altitude (m)': altitude_list,
        'Distance (m)': distance_list,
        'speed (m/s)': speed_list,
        'Acceleration (m/s^2)': acceleration_list
    }
    df = pd.DataFrame(data)
    df.to_csv('rocket_simulation_data.csv', index=False)
    return(apogee)
    
    #plt.figure(figsize=(10, 6))
    #plt.plot(time_list, altitude_list, label="Altitude", color="blue")
    #plt.plot(time_list, speed_list, label="speed", color="red")
    #plt.plot(time_list, acceleration_list, label="Acceleration", color="yellow")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Altitude (m), speed (m/s), Acceleration (m/s^2)")
    #plt.title("Rocket Altitude vs. Time")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

apogeeList = []

for i in range(41):
    cdAreaReal = 0.0173407967*0.35*(1+(0.1*i))
    apogeeList.append(simulate_rocket_flight(10000000, 0.01, 7048.7, 255.59, cdAreaReal))

for i in range(41):
    print(apogeeList[i])
