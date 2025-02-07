import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


def gravity_acceleration(h = 0,
                          phi = 31.04980169999999, # m/s^2
                          GM = 3.986004418e14, # 
                          a = 6378137
                         ):
    
    numerator = 1 + (0.00193185265245827352087 * (np.sin(phi) ** 2))
    denominator = np.sqrt(1 - 0.006694379990141316996137 * (np.sin(phi) ** 2))
    return 9.780325335903891718546 * (numerator/denominator) + (GM / ((a + h) ** 2)) - (GM / (a ** 2))

rho_f = lambda h: Atmosphere(h).density[0]

# create a function of time and state
def state_function(
    t,state,
    m = 25,
    CdA = 0.01
):
    x,z,vx,vz = state
    aero_force = CdA * rho_f(z) * np.sqrt(vx**2 + vz**2) / 2 / m
    ax = - vx * aero_force
    az = - gravity_acceleration(z) - aero_force * vz
    return(np.array([vx,vz,ax,az]))
    

state_function(0,[0,9144,.1,0])

def mach_event(t,state):
    x,z,vx,vz = state
    sound_speed = Atmosphere(z).speed_of_sound[0]
    speed = np.linalg.norm([vx,vz])
    return speed - 0.8*sound_speed
mach_event.terminal = True

def altitude_event(t,state):
    return state[1]
altitude_event.terminal = True


out = solve_ivp(
    state_function,
    t_span=(0,-55),
    y0=[0,9144+893,.1,0],
    max_step=1,
    events=[mach_event,altitude_event]
)

CdAs = np.linspace(0.01,0.05,500)
curves = {}
for CdA in CdAs:
    soln = solve_ivp(
        lambda t,y : state_function(t,y,CdA=CdA),
        t_span=(0,-55),
        y0=[0,9144+893,.1,0],
        max_step=1,
        events=[mach_event,altitude_event]
    )
    curves[CdA] = soln


cmap = lambda CdA: ((CdA - 0.01)/0.04,1.-(CdA - 0.01)/0.04,0)
for CdA,curve in curves.items():
    plt.plot(curve['y'][1],curve['y'][3],color=cmap(CdA))
plt.xlabel('Height (m)')
plt.ylabel('Velocity (m/s)')
plt.show()

velocities = np.concatenate([soln['y'][3] for soln in curves.values()])
heights = np.concatenate([soln['y'][1] for soln in curves.values()])
cdas = np.concatenate([np.full(soln['y'][1].shape,cda) for cda,soln in curves.items()])

def cost(x):
    best_guess = x[5] * velocities**2 + x[4] * heights**2 + x[3] * velocities * heights + x[2] * velocities +  x[1] * heights + x[0]
    loss = cdas - best_guess
    cost = np.dot(loss,loss)
    return cost

from scipy.optimize import minimize
optim = minimize(cost,[0.1,1e-3,1e-2,1e-1,1e-4,1e-6])

optim
