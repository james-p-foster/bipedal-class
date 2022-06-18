from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import isclose


def continuous_motion(t, x, m, g, l, gamma, alpha):
    theta, theta_dot = x
    return [theta_dot, (g / l) * np.sin(theta)]


def reset_map(x_minus, gamma, alpha):
    theta_minus, theta_dot_minus = x_minus
    theta_plus = gamma - alpha
    theta_dot_plus = theta_dot_minus * np.cos(2 * alpha)
    x_plus = [theta_plus, theta_dot_plus]
    return x_plus


def ground_strike_forward(t, x, gamma, alpha):
    theta, theta_dot = x
    return theta - (gamma + alpha)


def ground_strike_backward(t, x, gamma, alpha):
    theta, theta_dot = x
    return theta - (gamma - alpha)


def rimless_wheel_energy(m, g, l, theta, theta_dot):
    kinetic = 0.5 * m * l**2 * theta_dot**2
    potential = m * g * l * np.cos(theta)
    energy = kinetic + potential
    return energy


def find_required_initial_angular_velocity(g, l, gamma, alpha):
    return np.sqrt(2 * (g / l) * (1 - np.cos(gamma - alpha)))


def solve_rimless_wheel_trajectory(m, g, l, gamma, alpha, theta_dot_0, verbose=False):
    t_span = [0, 5]  # should be plenty long enough, events should interrupt before this finishes
    x0 = [gamma - alpha, theta_dot_0]

    # Need to do ground strike collision logic as a lambda because scipy only permits event functions to be in the form
    # event(t, x).
    ground_strike_forward_event = lambda t, x: ground_strike_forward(t, x, gamma, alpha)
    ground_strike_forward_event.terminal = True  # stops the ivp solver when event occurs
    ground_strike_forward_event.direction = 1  # event only triggers in forward direction, i.e. when theta comes from negative to equal gamma + alpha
    ground_strike_backward_event = lambda t, x: ground_strike_backward(t, x, gamma, alpha)
    ground_strike_backward_event.terminal = True
    ground_strike_backward_event.direction = -1  # event only triggers in backward direction, i.e. when theta comes from positive to equal gamma - alpha

    sol = solve_ivp(fun=lambda t, x: continuous_motion(t, x, m, g, l, gamma, alpha),
                    t_span=t_span, y0=x0, max_step=0.01,
                    events=(ground_strike_forward_event, ground_strike_backward_event),
                    dense_output=True)
    # verify that the correct event happened
    if x0[1] > find_required_initial_angular_velocity(g, l, gamma, alpha):
        assert isclose(sol.y_events[0][0][0], gamma + alpha)  # forward ground strike
    if x0[1] < find_required_initial_angular_velocity(g, l, gamma, alpha):
        assert isclose(sol.y_events[1][0][0], gamma - alpha)  # backward ground strike

    if verbose:
        print(f"Required angular velocity to complete step: {find_required_initial_angular_velocity(g, l, gamma, alpha)} rad/s.")
        print(f"gamma + alpha (= theta_0): {gamma + alpha}")
        print(f"gamma - alpha (= theta_f): {gamma - alpha}")
        # print(sol.y)
        # print(sol.t)
        # print(sol.t_events)
        # print(sol.y_events)
    return sol


def solve_rimless_wheel_trajectory_n_steps(m, g, l, gamma, alpha, theta_dot_0, num_steps=20):
    theta_dot_init = theta_dot_0
    for step in range(num_steps):
        sol = solve_rimless_wheel_trajectory(m, g, l, gamma, alpha, theta_dot_init, verbose=False)

        if step == 0:
            state_tape = sol.y
        else:
            state_tape = np.concatenate((state_tape, sol.y), axis=1)

        # Break out of loop if backward step is detected; nothing more will happen. This can be detected by seeing if
        # the first event detection array is empty -- this means a forward step hasn't been detected, so a
        # backward step must have occurred.
        if sol.y_events[0].size == 0:
            break

        x_minus = [sol.y[0][-1], sol.y[1][-1]]
        x_plus = reset_map(x_minus, gamma, alpha)
        theta_dot_init = x_plus[1]
    return state_tape


### PLOTTING FUNCTIONS ###


def find_axis_limits(state_tape, buffer=0.1):
    xlim = np.maximum(np.abs(np.min(state_tape[0])), np.abs(np.max(state_tape[0]))) + buffer
    xmin = -xlim
    xmax = xlim
    ylim = np.maximum(np.abs(np.min(state_tape[1])), np.abs(np.max(state_tape[1]))) + buffer
    ymin = -ylim
    ymax = ylim
    lims = [xmin, xmax, ymin, ymax]
    return lims
