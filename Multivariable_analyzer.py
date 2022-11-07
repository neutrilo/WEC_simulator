#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:15:01 2022

@author: albert
"""

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

# t = 0.0
# dt = 0.001
# tEnd = 10.0

y_pos_0 = 0.0  # Initial condition (Position)
y_vel_0 = 0.0  # Initial condition (Velocity)

z_old = np.array([y_pos_0, y_vel_0, y_pos_0, y_vel_0], float)
z_new = np.zeros(4, float)



# ---------------------------------------------------------------
# Solution
# ---------------------------------------------------------------
y1 = y_pos_0
v1 = y_vel_0
y2=y1
v2=v1
def make_data(y1,v1,y2,v2,m1,m2):
    t = 0.0
    dt = 0.001
    tEnd = 10.0
        
    g = 9.81  # Gravity [m/s^2]
    d = 1025  # Water density [kg/m^3]
    a = 0.1
    n = 100
    depth = 0
    H = np.linspace(-0.15, 0.15, n)  # Wave height [m]
    z_old = np.array([y1, v1, y2, v2], float)
    z_new = np.zeros(4, float)
    data = []
    data.append([t, y1, v1, y2, v2, 0, (m1 + m2) * g, 0])
    while t < tEnd:
        H = 0.2 * np.sin(2 * np.pi * t)  # Wave shape as sinus function
        pto = np.abs(damp * (z_old[1] - z_old[0]) * (z_old[3] - z_old[2]))  # Power Take Off (PTO)
        z_new = runge_kutta_step(rhs, z_old, dt)
        t += dt
        data.append([t, z_new[0], z_new[1], z_new[2], z_new[3], H, fbuoyancy(z_old[0], H), pto])
        z_old = z_new


data = make_data(y_pos_0, y_vel_0 , y_pos_0, y_vel_0,m1,m2)





















