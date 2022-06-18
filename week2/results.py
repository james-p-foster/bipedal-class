import numpy as np
import matplotlib.pyplot as plt

from week2.lipm import *

# System parameters
initial_state = [0.01, 0.1]
g = 9.81
z0 = 0.25

# First, let's consider the case where the stride_length and the top_of_stride_velocity for the next step are supplied.
# From this information, we can calculate the support_switch_position at which the next step begins.
stride_length = 1.0
top_of_stride_velocity = 0.5
state_tape = solve_lipm_n_steps(g, z0, initial_state, stride_length, top_of_stride_velocity, None)

buffer = 0.1
xlim = np.maximum(np.abs(np.min(state_tape[0])), np.abs(np.max(state_tape[0]))) + buffer
xmin = -xlim
xmax = xlim
ylim = np.maximum(np.abs(np.min(state_tape[1])), np.abs(np.max(state_tape[1]))) + buffer
ymin = -ylim
ymax = ylim

nx = 100
ny = 100
x_range = np.linspace(xmin, xmax, nx)
y_range = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_range, y_range)
energy = orbital_energy(X, Y, g, z0)

plt.scatter(state_tape[0][0], state_tape[1][0], marker='*')
plt.plot(state_tape[0], state_tape[1])
plt.title(f"Phase portrait of LIPM \n DESIRED: Stride length = {stride_length:.2f}m, Top of stride velocity = {top_of_stride_velocity:.2f}m/s")
plt.xlabel("CoM Position [m]")
plt.ylabel("CoM Velocity [m/s]")
plt.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
plt.hlines(top_of_stride_velocity, xmin, xmax, colors='k', linestyles='dashed', linewidths=0.5)
plt.vlines(np.max(state_tape[0]), ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
plt.vlines(np.min(state_tape[0]), ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
plt.savefig("gen/find_support_switch_position.png")
plt.show()

# Next, we'll consider the case when stride_length and the support_switch_position for the next step are supplied.
# From this information, we can calculate top_of_stride_velocity for the next step.
stride_length = 1.0
support_switch_position = 0.7
state_tape = solve_lipm_n_steps(g, z0, initial_state, stride_length, None, support_switch_position)

buffer = 0.1
xlim = np.maximum(np.abs(np.min(state_tape[0])), np.abs(np.max(state_tape[0]))) + buffer
xmin = -xlim
xmax = xlim
ylim = np.maximum(np.abs(np.min(state_tape[1])), np.abs(np.max(state_tape[1]))) + buffer
ymin = -ylim
ymax = ylim

nx = 100
ny = 100
x_range = np.linspace(xmin, xmax, nx)
y_range = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_range, y_range)
energy = orbital_energy(X, Y, g, z0)

plt.scatter(state_tape[0][0], state_tape[1][0], marker='*')
plt.plot(state_tape[0], state_tape[1])
plt.title(f"Phase portrait of LIPM \n DESIRED: Stride length = {stride_length:.2f}m, Support switch position = {support_switch_position:.2f}m")
plt.xlabel("CoM Position [m]")
plt.ylabel("CoM Velocity [m/s]")
plt.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
plt.vlines(np.max(state_tape[0]), ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
plt.vlines(np.min(state_tape[0]), ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
plt.savefig("gen/find_top_of_stride_velocity.png")
plt.show()

# Finally, we'll consider the case where top_of_stride_velocity and support_switch_position are supplied.
# From this information, we can calculate the stride_length of the next step..
top_of_stride_velocity = 1.0
support_switch_position = 0.3
state_tape = solve_lipm_n_steps(g, z0, initial_state, None, top_of_stride_velocity, support_switch_position, num_steps=50)

buffer = 0.1
xlim = np.maximum(np.abs(np.min(state_tape[0])), np.abs(np.max(state_tape[0]))) + buffer
xmin = -xlim
xmax = xlim
ylim = np.maximum(np.abs(np.min(state_tape[1])), np.abs(np.max(state_tape[1]))) + buffer
ymin = -ylim
ymax = ylim

nx = 100
ny = 100
x_range = np.linspace(xmin, xmax, nx)
y_range = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_range, y_range)
energy = orbital_energy(X, Y, g, z0)

plt.scatter(state_tape[0][0], state_tape[1][0], marker='*')
plt.plot(state_tape[0], state_tape[1])
plt.title(f"Phase portrait of LIPM \n DESIRED: Top of stride velocity = {top_of_stride_velocity:.2f}m/s, Support switch position = {support_switch_position:.2f}m")
plt.xlabel("CoM Position [m]")
plt.ylabel("CoM Velocity [m/s]")
plt.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
plt.hlines(top_of_stride_velocity, xmin, xmax, colors='k', linestyles='dashed', linewidths=0.5)
plt.vlines(np.max(state_tape[0]), ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
plt.savefig("gen/find_stride_length.png")
plt.show()
