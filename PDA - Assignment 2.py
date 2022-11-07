import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Definition
# ---------------------------------------------------------------

def fbuoyancy(depth, H):
    if depth - a / 2 > H:  # 1st Condition: Out of water
        force = 0
    elif depth + a / 2 < H:  # 2nd Condition: Fully submerged
        force = d * g * a ** 3
    else:  # 3rd Condition: Partly submerged
        force = d * g * (a / 2 - (depth - H)) * a ** 2
    return force


g = 9.81  # Gravity [m/s^2]
d = 1025  # Water density [kg/m^3]
a = 0.1
n = 100
depth = 0
H = np.linspace(-0.15, 0.15, n)  # Wave height [m]

def rhs(z_old, fbuoyancy): # We must reduce the order of the ODE
    return np.array([z_old[2], z_old[3], (1 / m1) * (fbuoyancy(z_old[0], H) - k * z_old[0] + k * (z_old[1] - z_old[0] - 0.5 * a) - m1 * g - damp * z_old[2]), (1 / m2) * (- k * (z_old[1] - z_old[0] - 0.5 * a) - m2 * g - damp * z_old[3])])

def runge_kutta_step(fun, z_old, dt):
    k1 = dt * fun(z_old, fbuoyancy)
    k2 = dt * fun(z_old + 0.5 * k1, fbuoyancy)
    k3 = dt * fun(z_old + 0.5 * k2, fbuoyancy)
    k4 = dt * fun(z_old + k3, fbuoyancy)
    return z_old + (1. / 6) * k1 + (2. / 6) * k2 + (2. / 6) * k3 + (1. / 6) * k4

# ---------------------------------------------------------------
# Pre-processing (Pseudo-Algorithm)
# ---------------------------------------------------------------

k = 20.0
damp = 0.1
m1 = 0.01 * d * g * a ** 3  # Body A [kg]
m2 = 0.2 * m1  # Body B [kg]

t = 0.0
dt = 0.001
tEnd = 10.0

y_pos_0 = 0.0  # Initial condition (Position)
y_vel_0 = 0.0  # Initial condition (Velocity)

z_old = np.array([y_pos_0, y_vel_0, y_pos_0, y_vel_0], float)
z_new = np.zeros(4, float)

data = []
data.append([t, y_pos_0, y_vel_0, y_pos_0, y_vel_0, 0, (m1 + m2) * g, 0])

# ---------------------------------------------------------------
# Solution
# ---------------------------------------------------------------

while t < tEnd:
    H = 0.2 * np.sin(2 * np.pi * t)  # Wave shape as sinus function
    pto = np.abs(damp * (z_old[1] - z_old[0]) * (z_old[3] - z_old[2]))  # Power Take Off (PTO)
    z_new = runge_kutta_step(rhs, z_old, dt)
    t += dt
    data.append([t, z_new[0], z_new[1], z_new[2], z_new[3], H, fbuoyancy(z_old[0], H), pto])
    z_old = z_new

# ---------------------------------------------------------------
#   Post-processing
# ---------------------------------------------------------------

data = np.array(data)
fig, ax = plt.subplots(5, 1, figsize=(14, 9))

# Wave Height
ax[0].plot(data[:, 0], data[:, 5], '-y', label='stokes wave')
ax[0].plot(data[:, 0], data[:, 1], '-r', label='position body A')
ax[0].plot(data[:, 0], data[:, 3], '-b', label='position body B')
# Buoyancy force
ax[1].plot(data[:, 0], data[:, 6], '-g', label='buoyancy force')
# Buoy position
ax[2].plot(data[:, 0], data[:, 2], '-k', label='buoy position')
# Differential velocity
ax[3].plot(data[:, 0], data[:, 4] - data[:, 3], '-m', label='differential velocity')
# Power Take Off (PTO)
ax[4].plot(data[:, 0], data[:, 7], '-r', label='power take off')

# Sketch plot
ax[0].set_ylabel('height [m]', fontsize=12)
ax[0].legend(loc='upper right')
ax[1].set_ylabel('force [N]', fontsize=12)
ax[1].legend(loc='upper right')
ax[2].set_ylabel('position [m]', fontsize=12)
ax[2].legend(loc='upper right')
ax[3].set_ylabel('velocity [m/s]', fontsize=12)
ax[3].legend(loc='upper right')
ax[4].set_xlabel('time [s]', fontsize=12)
ax[4].set_ylabel('energy [J]', fontsize=12)
ax[4].legend(loc='upper right')

plt.tight_layout()
plt.show()
