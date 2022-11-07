import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nx=100
t = 0.0
period = 2.0
tEnd = 2*period
dt = 0.01

x = np.linspace(0,4*np.pi,nx)
def my_sinus(x,t):
    return np.sin(x+t)

fig, ax = plt.subplots(figsize=(15,3))
plt.axis('equal')

line1, = ax.plot(x, my_sinus(x,0.0))

def init():
    ax.set_xlim([0, x[-1]])
    # ax.set_ylim([0, 1.0])
    return line1,


def animate(i):
    t = i * dt
    wave = my_sinus(x,t)
    line1.set_ydata(wave)
    return line1,

ani = FuncAnimation(fig,
                    func=animate,
                    init_func=init,
                    frames=int(tEnd / dt) + 1,
                    interval=int(dt * 1000),
                    blit=False,
                    repeat=False)
plt.show()