import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
M_E = 5.974e20
G = 6.6742e11
R_E = 6371e3  # radius of Earth (m)
Isp = 300  # specific impulse (s)
g0 = G*M_E/R_E**2 # 9.81  # gravitational acceleration at Earth's surface (m/s^2)
c = Isp * g0  # effective exhaust velocity (m/s)
A_e = 1.0  # exit area of the nozzle (m^2), assumed
p_a = 101325  # atmospheric pressure at sea level (Pa)
p_e = 0  # exhaust pressure, assumed vacuum for simplicity

# Initial conditions
v0 = 0  # initial velocity (m/s)
gamma0 = np.radians(90)  # initial flight path angle (radians), vertical launch
x0 = 0  # initial downrange distance (m)
h0 = 0  # initial altitude (m)
m_wet = 500e3  # initial wet mass (kg)
m_dry = 50e3  # dry mass (kg)
T = 7.5e6  # initial thrust (N)

# Time span
t = np.linspace(0, 3600, 1000)  # 3600 seconds, 1000 points

# Gravitational acceleration as a function of altitude
def gravity(h):
    return g0 * (R_E**2 / (R_E + h)**2)

# Differential equations
def rocket_dynamics(y, t, c, A_e, p_a, p_e, m_dry, T):
    v, gamma, x, h, m = y
    g = gravity(h)
    if m > m_dry:
        mdot_e = T / c  # mass flow rate (kg/s)
    else:
        mdot_e = 0  # no more propellant to burn
    T_dynamic = mdot_e * (c + (p_e - p_a) * A_e / mdot_e) if mdot_e != 0 else 0
    dvdt = (T_dynamic / m) - g * np.sin(gamma)
    if v == 0:
        print("v==0 at t=", t)
        dgamma_dt = 0
    else:
        print("v is", v)
        print("gamma is", gamma)
        dgamma_dt = -(1 / v) * (g - (v**2 / (R_E + h))) * np.cos(gamma)
    dxdt = (R_E / (R_E + h)) * v * np.cos(gamma)
    dhdt = v * np.sin(gamma)
    dmdt = -mdot_e  # rate of mass consumption
    return [dvdt, dgamma_dt, dxdt, dhdt, dmdt]

# Initial state
y0 = [v0, gamma0, x0, h0, m_wet]

# Integrate the equations over the time grid
solution = odeint(rocket_dynamics, y0, t, args=(c, A_e, p_a, p_e, m_dry, T))

# Extract the results
v, gamma, x, h, m = solution.T

# Plotting the trajectory
plt.figure(figsize=(10, 6))
plt.plot(x / 1e3, h / 1e3)  # convert to kilometers for readability
plt.title('Rocket Trajectory')
plt.xlabel('Downrange Distance (km)')
plt.ylabel('Altitude (km)')
plt.grid(True)
plt.show()

# Plotting g as a function of altitude
z = np.linspace(0, 1000e3, 1000)  # altitude from 0 to 1000 km
g_z = gravity(z)
plt.figure(figsize=(10, 6))
plt.plot(z / 1e3, g_z / g0)  # plot g/g0
plt.title('Gravitational Acceleration vs Altitude')
plt.xlabel('Altitude (km)')
plt.ylabel('g / g0')
plt.grid(True)
plt.show()

# Plotting mass vs time
plt.figure(figsize=(10, 6))
plt.plot(t, m / 1e3)  # convert mass to tonnes for readability
plt.title('Rocket Mass vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Mass (tonnes)')
plt.grid(True)
plt.show()

# Plotting velocity vs time
plt.figure(figsize=(10, 6))
plt.plot(t, v / 1e3)  # convert velocity to km/s for readability
plt.title('Rocket Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.grid(True)
plt.show()
