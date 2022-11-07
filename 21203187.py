import numpy as np

def my_stokes(amplitude, period, wave_length, h, x, t):
    omega = 2*np.pi / period
    k = 2* np.pi /wave_length
    sigma = np.tanh(k*h)
    theta = k*x-omega*t

    eta = amplitude * ( np.cos(theta) +\
        k*amplitude*(3.-sigma**2)/(4*sigma**3)*np.cos(2*theta))

    return eta

class My_Stokes_Class:
    def __init__(self):
        pass

    def set_para(self,amplitude, period, wave_length, h, x):
        self.amplitude = amplitude
        self.period = period
        self.wave_length = wave_length
        self.h  = h
        self.x = x
        self.omega = 2 * np.pi / self.period
        self.k = 2 * np.pi / self.wave_length
        self.sigma = np.tanh(self.k * self.h)

    def get_eta(self,t):
        self.theta = self.k * self.x - self.omega * t

        eta = self.amplitude * (np.cos(self.theta) + \
                           self.k * self.amplitude * (3. - self.sigma ** 2) / (4 * self.sigma ** 3) * np.cos(2 * self.theta))
        return eta
    """
    COFFEE BREAK TIL 10:22
    """