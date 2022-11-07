import numpy as np
import matplotlib.pyplot as plt
from my_methods import my_stokes
from my_methods import My_Stokes_Class

h = 0.8 #[m]
period = 1.0 #[s]
amplitude = 0.1 # [m]
wave_length = 2.0 #[m]
t = 0.0

x_max = 4.0
nx = 100
x = np.linspace(0,x_max,nx)

# eta = my_stokes(amplitude, period, wave_length, h, x, t)

wave = My_Stokes_Class()
wave.set_para(amplitude, period, wave_length, h, x)

eta = wave.get_eta(t)

plt.plot(x,eta)
plt.show()
