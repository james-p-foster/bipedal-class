import numpy as np
import matplotlib.pyplot as plt

from week1.rimless_wheel import *

# From Tedrake's example
m = 1.0  # > 0
g = 9.8  # > 0
l = 1.0  # > 0
gamma = 0.08  # 0 < gamma < pi/2. Because if >= pi/2 the slope would be past vertical
alpha = np.pi / 8  # 0 < alpha < pi/4. Because if >= pi/4, 2 * pi/4 = pi/2, all velocity would be cancelled on step

# First, let's plot the angular position and angular velocity over a single step. We'll do this for four initial
# angular velocities -- one way lower than required, one just under the required, one just over the required, and one
# way over the required.
theta_dot_0_array = [0, 0, 0, 0]
modifier = [-0.5, -0.01, 0.01, 0.5]
for i in range(4):
    theta_dot_0_array[i] = find_required_initial_angular_velocity(g, l, gamma, alpha) + modifier[i]
    sol = solve_rimless_wheel_trajectory(m, g, l, gamma, alpha, theta_dot_0_array[i])
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    fig.suptitle(f"Angular position and velocity of the rimless wheel \n theta_dot_0 = {theta_dot_0_array[i]:.2f}rad/s")
    ax1.plot(sol.t, sol.y[0])
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angular position [rad]")
    ax1.grid()
    ax2.plot(sol.t, sol.y[1])
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angular velocity [rad/s]")
    ax2.grid()
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(f"gen/angular_position_and_velocity{i}.png")
    plt.show()

# Second, let's see what happens to the phase portrait when we vary the slope of the ramp that the rimless wheel is on,
# while keeping all of the other parameters fixed.
gamma_samples = [0.08, 0.12, 0.16, 0.20]
fig = plt.figure()
fig.suptitle(f"Phase portrait of the rimless wheel \n with varying gamma")
for (i, gamma) in enumerate(gamma_samples):
    theta_dot_0 = find_required_initial_angular_velocity(g, l, gamma, alpha) * 2.0
    state_tape = solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0)
    ax = fig.add_subplot(2, 2, 1 + i)
    ax.scatter(state_tape[0][0], state_tape[1][0], marker='*')
    ax.plot(state_tape[0], state_tape[1])

    xmin, xmax, ymin, ymax = find_axis_limits(state_tape)

    nx = 100
    ny = 100
    x_range = np.linspace(xmin, xmax, nx)
    y_range = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_range, y_range)
    energy = rimless_wheel_energy(m, g, l, X, Y)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.vlines(gamma - alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.vlines(gamma + alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
    ax.set_title(f"{gamma:.2f}rad")
    ax.set_xlabel("Angular position [rad]")
    ax.set_ylabel("Angular velocity [rad/s]")
plt.tight_layout()
plt.savefig("gen/vary_gamma.png")
plt.show()

# Finally, let's create subplots to show how the phase portraits change when the parameters of the rimless wheel are
# varied. As theta_dotdot = (g / l) * sin(theta), the rimless wheel's continuous motion is unaffected by mass changes,
# only changes in g and l. The only other things that can affect the timing of the wheel's gait is touchdown times --
# given that these are triggered when theta = gamma + alpha, varying these two parameters can affect the phase portrait.
# We've already seen that varying the slope of the ramp (gamma) changes the phase portrait, so what about alpha? To
# summarise, we hypothesise that changes in the phase portrait can only be brought about varying g, l, gamma (already
# confirmed), and alpha... NO change will result from varying m. Let's test that.

# Vary m
m_samples = [0.1, 1.0, 5.0, 10.0]
fig = plt.figure()
fig.suptitle(f"Phase portrait of the rimless wheel \n with varying m")
for (i, m) in enumerate(m_samples):
    theta_dot_0 = find_required_initial_angular_velocity(g, l, gamma, alpha) * 2.0
    state_tape = solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0)
    ax = fig.add_subplot(2, 2, 1 + i)
    ax.scatter(state_tape[0][0], state_tape[1][0], marker='*')
    ax.plot(state_tape[0], state_tape[1])

    xmin, xmax, ymin, ymax = find_axis_limits(state_tape)

    nx = 100
    ny = 100
    x_range = np.linspace(xmin, xmax, nx)
    y_range = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_range, y_range)
    energy = rimless_wheel_energy(m, g, l, X, Y)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.vlines(gamma - alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.vlines(gamma + alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
    ax.set_title(f"{m:.2f}kg")
    ax.set_xlabel("Angular position [rad]")
    ax.set_ylabel("Angular velocity [rad/s]")
plt.tight_layout()
plt.savefig("gen/vary_m.png")
plt.show()

# Vary g
g_samples = [1.0, 5.0, 10.0, 20.0]
fig = plt.figure()
fig.suptitle(f"Phase portrait of the rimless wheel \n with varying g")
for (i, g) in enumerate(g_samples):
    theta_dot_0 = find_required_initial_angular_velocity(g, l, gamma, alpha) * 2.0
    state_tape = solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0)
    ax = fig.add_subplot(2, 2, 1 + i)
    ax.scatter(state_tape[0][0], state_tape[1][0], marker='*')
    ax.plot(state_tape[0], state_tape[1])

    xmin, xmax, ymin, ymax = find_axis_limits(state_tape)

    nx = 100
    ny = 100
    x_range = np.linspace(xmin, xmax, nx)
    y_range = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_range, y_range)
    energy = rimless_wheel_energy(m, g, l, X, Y)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.vlines(gamma - alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.vlines(gamma + alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
    ax.set_title(f"{g:.2f}m/s^2")
    ax.set_xlabel("Angular position [rad]")
    ax.set_ylabel("Angular velocity [rad/s]")
plt.tight_layout()
plt.savefig("gen/vary_g.png")
plt.show()

# Vary l
l_samples = [0.1, 1.0, 5.0, 10.0]
fig = plt.figure()
fig.suptitle(f"Phase portrait of the rimless wheel \n with varying l")
for (i, l) in enumerate(l_samples):
    theta_dot_0 = find_required_initial_angular_velocity(g, l, gamma, alpha) * 2.0
    state_tape = solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0)
    ax = fig.add_subplot(2, 2, 1 + i)
    ax.scatter(state_tape[0][0], state_tape[1][0], marker='*')
    ax.plot(state_tape[0], state_tape[1])

    xmin, xmax, ymin, ymax = find_axis_limits(state_tape)

    nx = 100
    ny = 100
    x_range = np.linspace(xmin, xmax, nx)
    y_range = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_range, y_range)
    energy = rimless_wheel_energy(m, g, l, X, Y)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.vlines(gamma - alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.vlines(gamma + alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
    ax.set_title(f"{l:.2f}m")
    ax.set_xlabel("Angular position [rad]")
    ax.set_ylabel("Angular velocity [rad/s]")
plt.tight_layout()
plt.savefig("gen/vary_l.png")
plt.show()

# Vary alpha
alpha_samples = [np.pi/14, np.pi/12, np.pi/10, np.pi/8]
fig = plt.figure()
fig.suptitle(f"Phase portrait of the rimless wheel \n with varying alpha")
for (i, alpha) in enumerate(alpha_samples):
    theta_dot_0 = find_required_initial_angular_velocity(g, l, gamma, alpha) * 2.0
    state_tape = solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0)
    ax = fig.add_subplot(2, 2, 1 + i)
    ax.scatter(state_tape[0][0], state_tape[1][0], marker='*')
    ax.plot(state_tape[0], state_tape[1])

    xmin, xmax, ymin, ymax = find_axis_limits(state_tape)

    nx = 100
    ny = 100
    x_range = np.linspace(xmin, xmax, nx)
    y_range = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_range, y_range)
    energy = rimless_wheel_energy(m, g, l, X, Y)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.vlines(gamma - alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.vlines(gamma + alpha, ymin, ymax, colors='k', linestyles='dashed', linewidths=0.5)
    ax.contour(X, Y, energy, levels=10, colors='k', linestyles='dotted', linewidths=0.25)
    ax.set_title(f"{alpha:.2f}rad")
    ax.set_xlabel("Angular position [rad]")
    ax.set_ylabel("Angular velocity [rad/s]")
plt.tight_layout()
plt.savefig("gen/vary_alpha.png")
plt.show()