from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def continuous_motion(t, state, g, z0):
    x, x_dot = state
    return [x_dot, (g / z0) * x]


def reset_map(state_minus, stride_length):
    x_minus, x_dot_minus = state_minus
    state_plus = [x_minus - stride_length, x_dot_minus]
    return state_plus


def step(t, state, x_final):
    x, x_dot = state
    return x - x_final

def solve_lipm_trajectory(state0, g, z0, x_final):
    t_span = [0, 5]  # should be plenty long enough

    step_event = lambda t, state: step(t, state, x_final)
    step_event.terminal = True

    sol = solve_ivp(fun=lambda t, state: continuous_motion(t, state, g, z0),
                    t_span=t_span, y0=state0, max_step=0.01,
                    events=step_event, dense_output=True)
    return sol


def orbital_energy(x, x_dot, g, z0):
    energy = 0.5 * x_dot**2 - 0.5 * (g / z0) * x**2
    return energy


def modify_support_switch_position(initial_state, g, z0, stride_length, top_of_stride_velocity):
    energy = orbital_energy(initial_state[0], initial_state[1], g, z0)
    next_energy = 0.5 * top_of_stride_velocity**2
    support_switch_position = (stride_length / 2) + (z0 / (g * stride_length)) * (next_energy - energy)
    return support_switch_position


def modify_top_of_stride_velocity(initial_state, g, z0, stride_length, support_switch_position):
    energy = orbital_energy(initial_state[0], initial_state[1], g, z0)
    next_energy = 0.5 * (g /z0) * (2 * support_switch_position * stride_length - stride_length**2) + energy
    top_of_stride_velocity = np.sqrt(2 * next_energy)
    return top_of_stride_velocity


def modify_stride_length(initial_state, g, z0, top_of_stride_velocity, support_switch_position):
    energy = orbital_energy(initial_state[0], initial_state[1], g, z0)
    constant_coeff = 2 * (0.5 * top_of_stride_velocity**2 - energy) * (z0 / g)
    poly_coeffs = np.array([1, -2*support_switch_position, constant_coeff])
    roots = np.roots(poly_coeffs)
    stride_length = np.max(roots)
    return stride_length


def solve_lipm_n_steps(g, z0, initial_state,
                       stride_length=None, top_of_stride_velocity=None, support_switch_position=None,
                       num_steps=5):
    missing_input_error = ValueError("Two out of three of stride_length, top_of_stride_velocity, and \
        support_switch_position must be supplied in order to solve for the final one.")
    if stride_length is None:
        if top_of_stride_velocity is None or support_switch_position is None:
            raise missing_input_error
        for i in range(num_steps):
            stride_length = modify_stride_length(initial_state, g, z0, top_of_stride_velocity, support_switch_position)
            sol = solve_lipm_trajectory(initial_state, g, z0, support_switch_position)
            if i == 0:
                state_tape = sol.y
            else:
                state_tape = np.concatenate((state_tape, sol.y), axis=1)
            state_minus = [sol.y[0][-1], sol.y[1][-1]]
            state_plus = reset_map(state_minus, stride_length)
            initial_state = state_plus

    if top_of_stride_velocity is None:
        if stride_length is None or support_switch_position is None:
            raise missing_input_error
        for i in range(num_steps):
            top_of_stride_velocity = modify_top_of_stride_velocity(initial_state, g, z0, stride_length, support_switch_position)
            sol = solve_lipm_trajectory(initial_state, g, z0, support_switch_position)
            if i == 0:
                state_tape = sol.y
            else:
                state_tape = np.concatenate((state_tape, sol.y), axis=1)
            state_minus = [sol.y[0][-1], sol.y[1][-1]]
            state_plus = reset_map(state_minus, stride_length)
            initial_state = state_plus

    if support_switch_position is None:
        if stride_length is None or top_of_stride_velocity is None:
            raise missing_input_error
        for i in range(num_steps):
            x_final = modify_support_switch_position(initial_state, g, z0, stride_length, top_of_stride_velocity)
            sol = solve_lipm_trajectory(initial_state, g, z0, x_final)
            if i == 0:
                state_tape = sol.y
            else:
                state_tape = np.concatenate((state_tape, sol.y), axis=1)
            state_minus = [sol.y[0][-1], sol.y[1][-1]]
            state_plus = reset_map(state_minus, stride_length)
            initial_state = state_plus
    return state_tape
