# main.py
import os
import time
from readsim import readsim
import numpy as np
import matplotlib.pyplot as plt

from rwsim import rwsim

os.system('clear')

simreader = readsim("./simulations/temp")
sim = rwsim()

sim.setReader(simreader)
sim.loadSim()
sim.alloc_particles()
sim.alloc_steps()
sim.particle_num = 10
sim.step_num = 20000

print(sim.step_size)

sim.step_size = 0.2
data = sim.eventloop()

print(data.shape)

x = data[:, 0:1000:, 0]
y = data[:, 0:1000:, 1]
z = data[:, 0:1000:, 2]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(sim.particle_num):
    ax.plot(x[i, :], y[i, :], z[i, :], marker="None", color=[
            np.random.rand(), np.random.rand(), np.random.rand()])

s = sim.swc[0, 4]
x0 = sim.swc[0, 1]
y0 = sim.swc[0, 2]
z0 = sim.swc[0, 3]
n = 1000
m = 1
theta = 2 * np.pi * np.random.rand(n, m)
v = np.random.rand(n, m)
phi = np.arccos((2 * v) - 1)
# r = np.power(np.random.rand(n, m), (1/3))
r = 1
x = x0 + s * r * np.sin(phi) * np.cos(theta)
y = y0 + s * r * np.sin(phi) * np.sin(theta)
z = z0 + s * r * np.cos(phi)


ax.scatter(x, y, z, marker=".")


s = sim.swc[1, 4]
x0 = sim.swc[1, 1]
y0 = sim.swc[1, 2]
z0 = sim.swc[1, 3]
n = 1000
m = 1
theta = 2 * np.pi * np.random.rand(n, m)
v = np.random.rand(n, m)
phi = np.arccos((2 * v) - 1)
# r = np.power(np.random.rand(n, m), (1/3))
r = 1
x = x0 + s * r * np.sin(phi) * np.cos(theta)
y = y0 + s * r * np.sin(phi) * np.sin(theta)
z = z0 + s * r * np.cos(phi)


ax.scatter(x, y, z, marker=".")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title(
    'Random Positions for Particle Init: npar = {}'.format(sim.particle_num))
plt.show()

print("DONE")

# st = time.time()
# particles = sim.init_particles()
# nd = time.time()-st
# print("Elapsed: {} seconds?".format(nd))
