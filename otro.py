#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:45:24 2022

@author: albert
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd





# ---------------------------------------------------------------
# Data exploring
# ---------------------------------------------------------------


exp_data = pd.read_csv("roll_decay_data.csv", delimiter= '\t')
# print(exp_data.shape)
# exp_data[exp_data.columns[0]][0]
data_t = np.array(exp_data[exp_data.columns[0]])
data_acel = np.array(exp_data[exp_data.columns[3]])
def integral(x,t,c=0):
    dt = t[1:]-t[0:-1]
    xmed = (x[1:]+x[0:-1])/2
    return np.append([c],dt*xmed)
data_vel = integral(data_acel,data_t) 
data_H = integral(data_vel,data_t)  


data_long = np.size(data_t)
spectrum = np.transpose(np.abs( np.fft.fft(data_acel)[1:int(data_long/4)] ))
spectrum = [np.linspace(2,len(spectrum)+1,len(spectrum))/np.max(data_t),spectrum/np.max(spectrum)]
mean = np.sum(spectrum[1]*spectrum[0])/np.sum(spectrum[1])
plt.axvline(x = mean, color = 'r', label = 'Frecuencia media')

plt.plot(spectrum[0],spectrum[1],label = 'FFT')
plt.legend()
plt.xlabel('Frecuencias (Hz)')
plt.ylabel('FFT normalizada inf')
plt.title('Aceleracion descompuesta en el dominio de frecuencias')
plt.show()





# ---------------------------------------------------------------
# Definition
# ---------------------------------------------------------------


g = 9.81  # Gravity [m/s^2]
d = 1025  # Water density [kg/m^3]
a = 0.1
n = 100
depth = 0
# H = np.linspace(-0.15, 0.15, n)  # Wave height [m]
H = np.interp(np.linspace(0,10,n),data_t,data_H)


k_base = 20.0
damp_base = 0.1
m1 = 0.01 * d * g * a ** 3  # Body A [kg]
m2 = 0.2 * m1  # Body B [kg]

t = 0.0
dt = 0.001
tEnd = 10.0
t_graf = np.transpose(np.linspace(0,tEnd,int(tEnd/dt) +1))

y_pos_0 = 0.0  # Initial condition (Position)
y_vel_0 = 0.0  # Initial condition (Velocity)

z_old = np.array([y_pos_0, y_vel_0, y_pos_0, y_vel_0], float)
z_base = z_old
z_new = np.zeros(4, float)




def make_data(z_old,m_mod1,m_mod2,kvar,damp):
    
    t = 0
    data = []
    data.append([t, y_pos_0, y_vel_0, y_pos_0, y_vel_0, 0, (m1 + m2) * g, 0])

    def fbuoyancy(depth, H):
        if depth - a / 2 > H:  # 1st Condition: Out of water
            force = 0
        elif depth + a / 2 < H:  # 2nd Condition: Fully submerged
            force = d * g * a ** 3
        else:  # 3rd Condition: Partly submerged
            force = d * g * (a / 2 - (depth - H)) * a ** 2
        return force
    
    
    
    
    def rhs(z_old, fbuoyancy): # We must reduce the order of the ODE
        return np.array([z_old[2], z_old[3], (1 / m_mod1) * (fbuoyancy(z_old[0], H) - kvar * z_old[0] + kvar * (z_old[1] - z_old[0] - 0.5 * a) - m_mod1 * g - damp * z_old[2]), (1 / m_mod2) * (- kvar * (z_old[1] - z_old[0] - 0.5 * a) - m_mod2 * g - damp * z_old[3])])
    
    def runge_kutta_step(fun, z_old, dt):
        k1 = dt * fun(z_old, fbuoyancy)
        k2 = dt * fun(z_old + 0.5 * k1, fbuoyancy)
        k3 = dt * fun(z_old + 0.5 * k2, fbuoyancy)
        k4 = dt * fun(z_old + k3, fbuoyancy)
        return z_old + (1. / 6) * k1 + (2. / 6) * k2 + (2. / 6) * k3 + (1. / 6) * k4
    
    # ---------------------------------------------------------------
    # Pre-processing (Pseudo-Algorithm)
    # ---------------------------------------------------------------
    
    
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
    
    return np.array(data)

def Position_1(t_graf,z_base,m_mod1,m_mod2,kvar,damp):
    dat = make_data(z_old,m_mod1,m_mod2,kvar)
    return np.interp(t_graf,dat[:,0],dat[:,1])
    

def Position_2(t_graf,z_base,m_mod1,m_mod2,kvar,damp):
    dat = make_data(z_old,m_mod1,m_mod2,kvar,damp)
    return np.interp(t_graf,dat[:,0],dat[:,3])

def Energy(t_graf,z_base,m_mod1,m_mod2,kvar,damp):
    dat = make_data(z_old,m_mod1,m_mod2,kvar,damp)
    return np.interp(t_graf,dat[:,0],dat[:,7])


# dat = make_data(z_old,m1,m2)


fig, ax = plt.subplots()
line, = ax.plot(t_graf, Energy(t_graf, z_base,m1,m2,k_base,damp_base), lw=2)
ax.set_xlabel('Time [s]')
plt.title('Energia generada')
fig.set_size_inches(18.5, 10.5, forward=True)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.5, bottom=0.15)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
m1_slider = Slider(
    ax=axfreq,
    label='Masa 1',
    valmin=0.5*m1,
    valmax=1.5*m1,
    valinit=m1,
    orientation="vertical"
)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
m2_slider = Slider(
    ax=axfreq,
    label='Masa 2',
    valmin=0.5*m2,
    valmax=1.5*m2,
    valinit=m2,
    orientation="vertical"
)


# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.3, 0.25, 0.0225, 0.63])
k_slider = Slider(
    ax=axamp,
    label="K",
    valmin=k_base*0.5,
    valmax=k_base*1.5,
    valinit=k_base,
    orientation="vertical"
)

# # Make a vertically oriented slider to control the amplitude
# axamp = fig.add_axes([0.4, 0.25, 0.0225, 0.63])
# damp_slider = Slider(
#     ax=axamp,
#     label="K",
#     valmin=damp_base*0.5 ,
#     valmax=damp_base*2,
#     valinit=damp_base,
#     orientation="vertical"
# )


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(Energy(t_graf, z_base, m1_slider.val,m2_slider.val, k_slider.val,damp_base))
    fig.canvas.draw_idle()
    
# register the update function with each slider
m1_slider.on_changed(update)
m2_slider.on_changed(update)
k_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.1, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    m1_slider.reset()
    m2_slider.reset()
    k_slider.reset()
button.on_clicked(reset)

plt.show()

# plt.plot(t_graf,Energy(t_graf,z_base,m1,m2,100))

# plt.plot(t_graf,Energy(t_graf,z_base,m1,m2,1))

# plt.show


